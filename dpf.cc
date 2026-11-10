#include "dpf.h"
#include "util.h"

#include <cstdlib>

namespace dpf {

namespace {

int bit_at(int value, int shift) {
  return (static_cast<std::uint32_t>(value) >> shift) & 1U;
}

int direction_at_level(int addr, std::size_t level, std::size_t lgn) {
  return bit_at(addr, static_cast<int>(lgn - level - 1));
}

} // namespace

word root_seed_share(prg& g, bool bit) {
  word root = g();
  root[0] = bit;
  return root;
}

word eval(const key& k, int addr) {
  const auto lgn = k.size() - 1;

  word seed = k[0];

  for (std::size_t i = 0; i < lgn; ++i) {
    const auto next = expand_node(seed, k[i + 1], true);
    const int dir = direction_at_level(addr, i, lgn);
    seed = next[dir];
  }

  return seed;
}
} // namespace dpf
