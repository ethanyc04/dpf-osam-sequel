#include "dpf.h"
#include "util.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace dpf {

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
} // namespace dpf
