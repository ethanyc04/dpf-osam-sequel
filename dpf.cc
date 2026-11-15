#include "dpf.h"
#include "util.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace dpf {

std::array<word, 2> expand_dpf_seed_uncorrected(
    word seed,
    std::size_t& prf_calls) {
  prf_calls += node_expansion_prf_calls;
  return expand_node_uncorrected_uncounted(seed);
}

std::array<word, 2> expand_dpf_seed(
    word seed,
    word correction,
    bool has_correction,
    std::size_t& prf_calls) {
  auto next = expand_dpf_seed_uncorrected(seed, prf_calls);
  if (has_correction && seed[0]) {
    next[0] ^= correction;
    next[1] ^= correction;
  }
  return next;
}

namespace {

std::size_t worker_threads() {
  const char* env = std::getenv("SAM_THREADS");
  if (env != nullptr) {
    const int requested = std::atoi(env);
    if (requested > 0) {
      return static_cast<std::size_t>(requested);
    }
  }

  return 1;
}

struct shift_task {
  int lgn;
  word seed;
  std::span<const word> corrections;
  int addr_diff;
  std::span<word> buf;
  std::size_t leaf_begin;
  std::size_t leaves;
};

void full_shift_mask_rec(
    int lgn,
    word seed,
    std::span<const word> corrections,
    int addr_diff,
    word val_diff,
    std::span<word> buf,
    std::size_t leaf_begin,
    std::size_t leaves,
    shift_mask_visitor visit,
    void* ctx,
    std::size_t& prf_calls) {
  if (leaves == 1) {
    word point_share = seed;
    if (seed[0]) {
      point_share ^= val_diff;
    }
    if (visit == nullptr) {
      buf[0] ^= point_share;
    } else {
      visit(leaf_begin, point_share, ctx);
    }
    return;
  }

  auto next = expand_dpf_seed(
      seed,
      corrections[0],
      true,
      prf_calls);

  const int dir = bit_at(addr_diff, lgn - 1);
  if (dir) {
    std::swap(next[0], next[1]);
  }

  const std::size_t half = leaves / 2;
  const bool visiting = visit != nullptr;
  full_shift_mask_rec(
      lgn - 1,
      next[0],
      corrections.subspan(1),
      addr_diff,
      val_diff,
      visiting ? std::span<word>{} : buf.subspan(0, half),
      leaf_begin,
      half,
      visit,
      ctx,
      prf_calls);
  full_shift_mask_rec(
      lgn - 1,
      next[1],
      corrections.subspan(1),
      addr_diff,
      val_diff,
      visiting ? std::span<word>{} : buf.subspan(half),
      leaf_begin + half,
      leaves - half,
      visit,
      ctx,
      prf_calls);
}

void make_shift_tasks(
    int lgn,
    word seed,
    std::span<const word> corrections,
    int addr_diff,
    std::span<word> buf,
    std::size_t leaf_begin,
    std::size_t leaves,
    std::size_t task_count,
    std::vector<shift_task>& tasks,
    std::size_t& prf_calls) {
  if (task_count <= 1 || leaves == 1) {
    tasks.push_back({
      .lgn = lgn,
      .seed = seed,
      .corrections = corrections,
      .addr_diff = addr_diff,
      .buf = buf,
      .leaf_begin = leaf_begin,
      .leaves = leaves,
    });
    return;
  }

  auto next = expand_dpf_seed(
      seed,
      corrections[0],
      true,
      prf_calls);

  const int dir = bit_at(addr_diff, lgn - 1);
  if (dir) {
    std::swap(next[0], next[1]);
  }

  const std::size_t half = leaves / 2;
  const std::size_t left_tasks = task_count / 2;
  const std::size_t right_tasks = task_count - left_tasks;
  const bool visiting = buf.empty();
  make_shift_tasks(
      lgn - 1,
      next[0],
      corrections.subspan(1),
      addr_diff,
      visiting ? std::span<word>{} : buf.subspan(0, half),
      leaf_begin,
      half,
      left_tasks,
      tasks,
      prf_calls);
  make_shift_tasks(
      lgn - 1,
      next[1],
      corrections.subspan(1),
      addr_diff,
      visiting ? std::span<word>{} : buf.subspan(half),
      leaf_begin + half,
      leaves - half,
      right_tasks,
      tasks,
      prf_calls);
}

class worker_pool {
public:
  explicit worker_pool(std::size_t worker_count) {
    threads_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
      threads_.emplace_back([this] { worker_loop(); });
    }
  }

  ~worker_pool() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    work_ready_.notify_all();
    for (auto& worker : threads_) {
      worker.join();
    }
  }

  void run(
      const std::vector<shift_task>& tasks,
      word val_diff,
      shift_mask_visitor visit,
      void* ctx,
      std::vector<std::size_t>& prf_calls) {
    if (tasks.empty()) {
      return;
    }
    std::lock_guard run_lock(run_mutex_);
    {
      std::lock_guard lock(mutex_);
      tasks_ = &tasks;
      val_diff_ = val_diff;
      visit_ = visit;
      visit_ctx_ = ctx;
      prf_calls_ = &prf_calls;
      next_task_ = 0;
      remaining_ = tasks.size();
    }
    work_ready_.notify_all();

    std::unique_lock lock(mutex_);
    work_done_.wait(lock, [&] {
      return remaining_ == 0;
    });
    tasks_ = nullptr;
    visit_ = nullptr;
    visit_ctx_ = nullptr;
    prf_calls_ = nullptr;
  }

private:
  void worker_loop() {
    while (true) {
      std::size_t task_index = 0;
      const std::vector<shift_task>* tasks = nullptr;
      std::vector<std::size_t>* prf_calls = nullptr;
      shift_mask_visitor visit = nullptr;
      void* visit_ctx = nullptr;
      word val_diff;
      {
        std::unique_lock lock(mutex_);
        work_ready_.wait(lock, [&] {
          return stopping_
            || (tasks_ != nullptr && next_task_ < tasks_->size());
        });
        if (stopping_) {
          return;
        }
        task_index = next_task_++;
        tasks = tasks_;
        prf_calls = prf_calls_;
        visit = visit_;
        visit_ctx = visit_ctx_;
        val_diff = val_diff_;
      }

      std::size_t local_prf_calls = 0;
      const auto& task = (*tasks)[task_index];
      full_shift_mask_rec(
          task.lgn,
          task.seed,
          task.corrections,
          task.addr_diff,
          val_diff,
          task.buf,
          task.leaf_begin,
          task.leaves,
          visit,
          visit_ctx,
          local_prf_calls);
      (*prf_calls)[task_index] = local_prf_calls;

      {
        std::lock_guard lock(mutex_);
        --remaining_;
        if (remaining_ == 0) {
          work_done_.notify_one();
        }
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex run_mutex_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  bool stopping_ = false;
  std::size_t next_task_ = 0;
  std::size_t remaining_ = 0;
  const std::vector<shift_task>* tasks_ = nullptr;
  std::vector<std::size_t>* prf_calls_ = nullptr;
  shift_mask_visitor visit_ = nullptr;
  void* visit_ctx_ = nullptr;
  word val_diff_;
};

worker_pool& eval_worker_pool() {
  static worker_pool pool(worker_threads());
  return pool;
}

void run_full_shift_mask(
    const key& k,
    int addr_diff,
    word val_diff,
    std::span<word> buf,
    shift_mask_visitor visit,
    void* ctx) {
  if (k.size() == 0) {
    throw std::invalid_argument("Cannot evaluate an empty DPF key");
  }
  const auto depth = k.size() - 1;
  if (depth > static_cast<std::size_t>(std::numeric_limits<int>::digits)
      || depth >= std::numeric_limits<std::size_t>::digits) {
    throw std::invalid_argument("DPF depth exceeds supported address range");
  }
  const auto lgn = static_cast<int>(depth);
  const auto leaves = std::size_t{1} << lgn;
  if (addr_diff < 0 || static_cast<std::size_t>(addr_diff) >= leaves) {
    throw std::invalid_argument("DPF address shift outside domain");
  }
  if (visit == nullptr && buf.size() != leaves) {
    throw std::invalid_argument("DPF shift output span size mismatch");
  }

  std::span<const word> corrections(k.data(), k.size());
  const std::size_t task_count = std::min(worker_threads(), leaves);

  if (task_count < 2) {
    std::size_t prf_calls = 0;
    full_shift_mask_rec(
        lgn,
        k[0],
        corrections.subspan(1),
        addr_diff,
        val_diff,
        buf,
        0,
        leaves,
        visit,
        ctx,
        prf_calls);
    prf::add_calls(prf_calls);
    return;
  }

  std::size_t split_prf_calls = 0;
  std::vector<shift_task> tasks;
  tasks.reserve(task_count);
  make_shift_tasks(
      lgn,
      k[0],
      corrections.subspan(1),
      addr_diff,
      buf,
      0,
      leaves,
      task_count,
      tasks,
      split_prf_calls);

  std::size_t prf_calls = split_prf_calls;
  std::vector<std::size_t> worker_prf_calls(tasks.size());
  eval_worker_pool().run(tasks, val_diff, visit, ctx, worker_prf_calls);
  for (const auto calls : worker_prf_calls) {
    prf_calls += calls;
  }
  prf::add_calls(prf_calls);
}

} // namespace

word root_seed_share(prg& g, bool root_bit) {
  word root = g();
  root[0] = root_bit;
  return root;
}

word eval(const key& k, int addr) {
  if (k.size() == 0) {
    throw std::invalid_argument("Cannot evaluate an empty DPF key");
  }
  const auto lgn = k.size() - 1;
  if (lgn > static_cast<std::size_t>(std::numeric_limits<int>::digits)) {
    throw std::invalid_argument("DPF depth exceeds non-negative int address range");
  }
  if (addr < 0 || static_cast<std::uint64_t>(addr) >= (std::uint64_t{1} << lgn)) {
    throw std::invalid_argument("DPF address outside domain");
  }

  word seed = k[0];

  for (std::size_t i = 0; i < lgn; ++i) {
    const auto next = expand_node(seed, k[i + 1], true);
    const int dir = direction_at_level(addr, i, lgn);
    seed = next[dir];
  }

  return seed;
}

void full_shift_mask_into(
    const key& k,
    int addr_diff,
    word val_diff,
    std::span<word> buf) {
  run_full_shift_mask(k, addr_diff, val_diff, buf, nullptr, nullptr);
}

void full_shift_mask_visit(
    const key& k,
    int addr_diff,
    word val_diff,
    shift_mask_visitor visit,
    void* ctx) {
  if (visit == nullptr) {
    throw std::invalid_argument("DPF shift visitor cannot be null");
  }

  run_full_shift_mask(k, addr_diff, val_diff, {}, visit, ctx);
}

} // namespace dpf
