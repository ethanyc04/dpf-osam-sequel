#include "backend.h"

#include <array>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace {

thread_local int id = -1;

template <typename T>
struct inbox {
  std::mutex mutex;
  std::condition_variable ready;
  std::queue<T> values;
};

std::array<inbox<int>, 2> addresses;
std::array<inbox<word>, 2> words;

template <typename T>
void send(int dst_party, T value, std::array<inbox<T>, 2>& queues) {
  if (dst_party != 1 - party_id()) {
    throw std::invalid_argument("Invalid local send destination");
  }
  auto& queue = queues[dst_party];
  {
    std::lock_guard lock(queue.mutex);
    queue.values.push(value);
  }
  queue.ready.notify_one();
}

template <typename T>
T receive(int src_party, std::array<inbox<T>, 2>& queues) {
  const int own_party = party_id();
  if (src_party != 1 - own_party) {
    throw std::invalid_argument("Invalid local receive source");
  }
  auto& queue = queues[own_party];
  std::unique_lock lock(queue.mutex);
  queue.ready.wait(lock, [&] { return !queue.values.empty(); });
  const T value = queue.values.front();
  queue.values.pop();
  return value;
}

} // namespace

void set_party_id(int party) {
  if (party != 0 && party != 1) {
    throw std::invalid_argument("Party ID must be 0 or 1");
  }
  id = party;
}

int party_id() {
  if (id < 0) throw std::logic_error("Local party ID has not been set");
  return id;
}

const char* backend_name() { return "local"; }

void send_addr(int x) { send(1 - party_id(), x, addresses); }
int recv_addr() { return receive(1 - party_id(), addresses); }
void send_word(int dst_party, word x) { send(dst_party, x, words); }
word recv_word(int src_party) { return receive(src_party, words); }
