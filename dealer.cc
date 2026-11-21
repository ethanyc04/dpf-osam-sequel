#include "preprocessing.h"
#include "full_eval_sam.h"
#include "reactive_sam.h"
#include "dealer_shuffle.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>

namespace {

std::array<std::vector<int>, 2> sample_permutation_shares(int n) {
  // Fisher-Yates with unbiased bounded draws, then fresh shares of each entry.
  std::array<std::vector<int>, 2> out;
  std::vector<int> permutation(n);
  std::iota(permutation.begin(), permutation.end(), 0);
  prg g;
  for (int remaining = n; remaining > 1; --remaining) {
    std::swap(permutation[remaining - 1], permutation[g.draw_int(remaining)]);
  }
  for (int i = 0; i < n; ++i) {
    const int first_share = g.draw_int(n);
    out[0].push_back(first_share);
    out[1].push_back(permutation[i] ^ first_share);
  }
  return out;
}

} // namespace

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

  auto shares = sample_permutation_shares(capacity);
  out[0].pi = std::move(shares[0]);
  out[1].pi = std::move(shares[1]);

  const auto depth = static_cast<std::size_t>(lg(capacity));
  for (int i = 0; i < max_writes; ++i) {
    auto correlations = dpf_corr::preprocess_dealer(depth);
    out[0].write_corrs.push_back(std::move(correlations[0]));
    out[1].write_corrs.push_back(std::move(correlations[1]));
  }
  return out;
}

} // namespace sam::full_eval

namespace sam::reactive {

std::array<resources, 2> preprocess_dealer(
    int capacity, int max_dpf_corrs, int value_bits) {
  if (!is_power_of_two(capacity)) {
    throw std::invalid_argument("SAM capacity must be a positive power of two");
  }
  if (max_dpf_corrs < 0) {
    throw std::invalid_argument("DPF correlation budget must be non-negative");
  }
  check_value_bits(value_bits);

  std::array<resources, 2> out;
  for (auto& r : out) {
    r.storage_size = capacity;
    r.value_bits = value_bits;
    r.pi.resize(capacity);
    r.dpf_corrs.reserve(max_dpf_corrs);
  }

  auto shares = sample_permutation_shares(capacity);
  out[0].pi = std::move(shares[0]);
  out[1].pi = std::move(shares[1]);

  const auto depth = static_cast<std::size_t>(lg(capacity));
  for (int i = 0; i < max_dpf_corrs; ++i) {
    auto correlations = dpf_corr::preprocess_dealer(depth);
    out[0].dpf_corrs.push_back(std::move(correlations[0]));
    out[1].dpf_corrs.push_back(std::move(correlations[1]));
  }
  return out;
}

} // namespace sam::reactive  

namespace dealer_shuffle {

std::pair<p0_state, p1_state> preprocess_dealer(
    std::size_t slot_count, std::size_t slot_bits) {
  if (slot_bits == 0) {
    throw std::invalid_argument("Shuffle slot width must be positive");
  }
  if (slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Shuffle slot count exceeds supported permutation range");
  }
  const std::size_t words_per_slot = slot_bits / word_bits + (slot_bits % word_bits != 0);
  p0_state p0;
  p1_state p1;
  if (slot_count > p0.s.max_size() / words_per_slot) {
    throw std::length_error("Shuffle mask size exceeds supported vector size");
  }
  if (slot_count != 0) {
    prg g;
    p0.seed = g();
    p1.seed = g();
  }
  prepare_p0(p0, slot_count, slot_bits);
  prepare_p1(p1, slot_count, slot_bits);

  auto r1 = permute(p0.pi0, p1.r, words_per_slot);
  for (std::size_t i = 0; i < r1.size(); ++i) {
    r1[i] ^= p0.s[i];
  }
  p0.r2 = permute(p1.pi1, r1, words_per_slot);
  for (std::size_t i = 0; i < p0.r2.size(); ++i) {
    p0.r2[i] ^= p1.t[i];
  }
  return {std::move(p0), std::move(p1)};
}

} // namespace dealer_shuffle
