#include "dpf.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <thread>
#include <vector>

namespace dpf {

namespace {

word advance_level(std::array<word, 2>& seeds, int direction, std::size_t& prf_calls) {
  auto children0 = detail::expand_seed_uncorrected(seeds[0], prf_calls);
  auto children1 = detail::expand_seed_uncorrected(seeds[1], prf_calls);
  const auto inactive = direction ^ 1U;
  const word correction = children0[inactive] ^ children1[inactive];
  if (seeds[0][0]) children0[direction] ^= correction;
  if (seeds[1][0]) children1[direction] ^= correction;
  seeds = { children0[direction], children1[direction] };
  return correction;
}

struct block_range {
  std::size_t begin;
  std::size_t end;
};

std::vector<block_range> make_blocks(std::size_t n) {
  if (n == 0) {
    return {};
  }
  const std::size_t workers = std::min(detail::worker_threads(), n);
  std::vector<block_range> blocks;
  blocks.reserve(workers);
  const std::size_t chunk = n / workers;
  const std::size_t extra = n % workers;
  std::size_t begin = 0;
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t end = begin + chunk + (worker + 1 == workers ? extra : 0);
    blocks.push_back({ begin, end });
    begin = end;
  }
  return blocks;
}

} // namespace

std::array<std::vector<key>, 2> generate_dealer_batch(
    std::size_t lgn, const std::vector<int>& addresses) {
  const auto domain = detail::domain_size(lgn);
  for (const int addr : addresses) {
    detail::validate_address(addr, domain);
  }

  const std::size_t batch = addresses.size();
  if (batch == 0) {
    return {};
  }

  std::array<std::vector<key>, 2> keys {
    std::vector<key>(batch, key(lgn + 1)),
    std::vector<key>(batch, key(lgn + 1)),
  };
  std::vector<std::array<word, 2>> seeds(batch);

  // Only the calling thread accesses the PRG. Workers receive fixed randomness.
  prg root_rng;
  for (std::size_t j = 0; j < batch; ++j) {
    keys[0][j][0] = detail::root_seed_share(root_rng, false);
    keys[1][j][0] = detail::root_seed_share(root_rng, true);
    seeds[j] = { keys[0][j][0], keys[1][j][0] };
    keys[0][j].mask_share = root_rng();
  }
  if (lgn == 0) {
    for (std::size_t j = 0; j < batch; ++j) {
      keys[1][j].mask_share = keys[0][j].mask_share ^ seeds[j][0] ^ seeds[j][1];
    }
    return keys;
  }

  const auto blocks = make_blocks(batch);
  std::vector<std::size_t> worker_prf_calls(blocks.size());
  const auto generate_block = [&](std::size_t worker) {
    const auto block = blocks[worker];
    std::size_t local_prf_calls = 0;
    for (std::size_t level = 0; level < lgn; ++level) {
      for (std::size_t j = block.begin; j < block.end; ++j) {
        const auto direction = detail::direction_at_level(addresses[j], level, lgn);
        const word correction = advance_level(seeds[j], direction, local_prf_calls);
        keys[0][j][level + 1] = correction;
        keys[1][j][level + 1] = correction;
      }
    }
    for (std::size_t j = block.begin; j < block.end; ++j) {
      // Freshly share the output mask without exposing either target-leaf seed.
      keys[1][j].mask_share = keys[0][j].mask_share ^ seeds[j][0] ^ seeds[j][1];
    }
    worker_prf_calls[worker] = local_prf_calls;
  };

  if (blocks.size() == 1) {
    generate_block(0);
  } else {
    std::vector<std::thread> threads;
    threads.reserve(blocks.size());
    try {
      for (std::size_t worker = 0; worker < blocks.size(); ++worker) {
        threads.emplace_back(generate_block, worker);
      }
    } catch (...) {
      for (auto& worker : threads) worker.join();
      throw;
    }
    for (auto& worker : threads) worker.join();
  }
  std::size_t aggregate_prf_calls = 0;
  for (const auto calls : worker_prf_calls) aggregate_prf_calls += calls;
  prf::add_calls(aggregate_prf_calls);
  return keys;
}

std::array<key, 2> generate_dealer(std::size_t lgn, int addr) {
  const auto domain = detail::domain_size(lgn);
  detail::validate_address(addr, domain);

  std::array<key, 2> k { key(lgn + 1), key(lgn + 1) };
  prg root_rng;

  k[0][0] = detail::root_seed_share(root_rng, false);
  k[1][0] = detail::root_seed_share(root_rng, true);
  std::array<word, 2> seeds { k[0][0], k[1][0] };
  std::size_t aggregate_prf_calls = 0;

  for (std::size_t level = 0; level < lgn; ++level) {
    const auto direction = detail::direction_at_level(addr, level, lgn);
    const word correction = advance_level(seeds, direction, aggregate_prf_calls);
    k[0][level + 1] = correction;
    k[1][level + 1] = correction;
  }
  prf::add_calls(aggregate_prf_calls);

  // Freshly share the output mask instead of exposing either target-leaf seed.
  const word output_mask = seeds[0] ^ seeds[1];
  k[0].mask_share = root_rng();
  k[1].mask_share = k[0].mask_share ^ output_mask;
  return k;
}

} // namespace dpf
