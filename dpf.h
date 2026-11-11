#ifndef DPF_H__
#define DPF_H__

#include "types.h"
#include "util.h"

#include <array>
#include <cstddef>
#include <vector>

namespace dpf {

using ::word;

inline constexpr std::size_t node_expansion_prf_calls = 2;

inline std::size_t ceil_div(std::size_t numerator, std::size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

inline word dpf_domain_separator(int domain) {
  word out;
  out[0] = (domain & 1) != 0;
  out[word_bits / 2] = (domain & 2) != 0;
  out[word_bits - 1] = true;
  return out;
}

inline const word dpf_domain_separator_0 = dpf_domain_separator(0);
inline const word dpf_domain_separator_1 = dpf_domain_separator(1);
inline const word dpf_domain_separator_2 = dpf_domain_separator(2);
inline const word dpf_domain_separator_3 = dpf_domain_separator(3);

inline const word& dpf_domain_separator_cached(int domain) {
  switch (domain) {
    case 0: return dpf_domain_separator_0;
    case 1: return dpf_domain_separator_1;
    case 2: return dpf_domain_separator_2;
    case 3: return dpf_domain_separator_3;
    default: throw std::invalid_argument("DPF domain separator out of range");
  }
}

inline const prf& fixed_dpf_prf() {
  static const prf f(dpf_domain_separator_cached(3));
  return f;
}

inline std::array<word, 2> expand_node_uncorrected_uncounted(word seed) {
  const bool bit = seed[0];
  const word input0 = seed ^ dpf_domain_separator_cached(0);
  const word input1 = seed ^ dpf_domain_separator_cached(1);
  auto next = fixed_dpf_prf().eval_pair(input0, input1);
  next[0] ^= input0;
  next[1] ^= input1;
  next[1][0] = next[0][0] ^ bit;
  return next;
}

inline std::array<word, 2> expand_node_uncorrected(word seed) {
  prf::add_calls(node_expansion_prf_calls);
  return expand_node_uncorrected_uncounted(seed);
}

inline std::array<word, 2> expand_node(
    word seed,
    word correction,
    bool has_correction) {
  auto next = expand_node_uncorrected(seed);
  if (has_correction && seed[0]) {
    next[0] ^= correction;
    next[1] ^= correction;
  }
  return next;
}

struct key {
  std::vector<word> words;
  word mask_share; // Share of the DPF's random output value, used for derandomization.

  key() = default;
  explicit key(std::size_t n) : words(n) { } // Size n yields DPF with domain size 2^(n-1)

  std::size_t size() const { return words.size(); }
  word* data() { return words.data(); }
  const word* data() const { return words.data(); }
  word& operator[](std::size_t i) { return words[i]; }
  const word& operator[](std::size_t i) const { return words[i]; }
};

inline int bit_at(int value, int shift) {
  return (static_cast<std::uint32_t>(value) >> shift) & 1U;
}

inline int direction_at_level(int addr, std::size_t level, std::size_t lgn) {
  return bit_at(addr, static_cast<int>(lgn - level - 1));
}

word root_seed_share(prg& g, bool root_bit);

std::array<key, 2> generate_dealer(std::size_t lgn, int addr);

word eval(const key&, int addr);

} // namespace dpf

#endif
