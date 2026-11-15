#include "dpf.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace dpf {

std::array<key, 2> generate_dealer(std::size_t lgn, int addr) {
  if (lgn > static_cast<std::size_t>(std::numeric_limits<int>::digits)) {
    throw std::invalid_argument("DPF depth exceeds non-negative int address range");
  }
  if (addr < 0 || static_cast<std::uint64_t>(addr) >= (std::uint64_t{1} << lgn)) {
    throw std::invalid_argument("DPF address outside domain");
  }

  std::array<key, 2> k { key(lgn + 1), key(lgn + 1) };
  prg root_rng;

  k[0][0] = root_seed_share(root_rng, false);
  k[1][0] = root_seed_share(root_rng, true);
  std::array<word, 2> seeds { k[0][0], k[1][0] };
  std::size_t aggregate_prf_calls = 0;

  for (std::size_t level = 0; level < lgn; ++level) {
    auto children0 = expand_dpf_seed_uncorrected(seeds[0], aggregate_prf_calls);
    auto children1 = expand_dpf_seed_uncorrected(seeds[1], aggregate_prf_calls);
    const auto direction = direction_at_level(addr, level, lgn);
    const auto inactive = direction ^ 1U;
    const word correction = children0[inactive] ^ children1[inactive];
    k[0][level + 1] = correction;
    k[1][level + 1] = correction;

    if (seeds[0][0]) {
      children0[direction] ^= correction;
    }
    if (seeds[1][0]) {
      children1[direction] ^= correction;
    }
    seeds[0] = children0[direction];
    seeds[1] = children1[direction];
  }
  prf::add_calls(aggregate_prf_calls);

  // Freshly share the output mask instead of exposing either target-leaf seed.
  const word output_mask = seeds[0] ^ seeds[1];
  k[0].mask_share = root_rng();
  k[1].mask_share = k[0].mask_share ^ output_mask;
  return k;
}

} // namespace dpf
