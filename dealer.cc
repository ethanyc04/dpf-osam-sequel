#include "preprocessing.h"
#include "full_eval_sam.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>

namespace dpf_corr {

std::array<correlation, 2> preprocess_dealer(std::size_t lgn) {
  const auto domain = dpf::detail::domain_size(lgn);

  prg g;
  const auto mask = domain - 1;
  const int alpha = static_cast<int>(g.draw_u32() & mask);
  auto keys = dpf::generate_dealer(lgn, alpha);
  const int alpha1 = static_cast<int>(g.draw_u32() & mask);
  const int alpha2 = alpha ^ alpha1;

  return { correlation { alpha1, std::move(keys[0]) },
           correlation { alpha2, std::move(keys[1]) } };
}

} // namespace dpf_corr

namespace sam::full_eval {

std::array<resources, 2> preprocess_dealer(int capacity, int max_writes, int value_bits) {
  if (!is_power_of_two(capacity)) {
    throw std::invalid_argument("SAM capacity must be a positive power of two");
  }
  if (max_writes < 0) {
    throw std::invalid_argument("SAM write budget must be non-negative");
  }
  check_value_bits(value_bits);

  std::array<resources, 2> out;
  for (auto& r : out) {
    r.storage_size = capacity;
    r.value_bits = value_bits;
    r.pi.resize(capacity);
    r.write_corrs.reserve(max_writes);
  }

  // Fisher-Yates with unbiased bounded draws, then fresh shares of each entry.
  auto& permutation = out[0].pi;
  std::iota(permutation.begin(), permutation.end(), 0);
  prg g;
  for (int remaining = capacity; remaining > 1; --remaining) {
    std::swap(permutation[remaining - 1], permutation[g.draw_int(remaining)]);
  }
  for (int i = 0; i < capacity; ++i) {
    const int first_share = g.draw_int(capacity);
    out[1].pi[i] = permutation[i] ^ first_share;
    permutation[i] = first_share;
  }

  const auto depth = static_cast<std::size_t>(lg(capacity));
  for (int i = 0; i < max_writes; ++i) {
    auto correlations = dpf_corr::preprocess_dealer(depth);
    out[0].write_corrs.push_back(std::move(correlations[0]));
    out[1].write_corrs.push_back(std::move(correlations[1]));
  }
  return out;
}

} // namespace sam::full_eval
