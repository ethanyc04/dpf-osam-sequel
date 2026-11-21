#ifndef DEALER_SHUFFLE_H
#define DEALER_SHUFFLE_H

#include "util.h"
#include "backend.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dealer_shuffle {

using ::word;

namespace {

// Fisher-Yates
inline std::vector<int> sample_permutation(std::size_t n, prg& g) {
  std::vector<int> pi(n);
  std::iota(pi.begin(), pi.end(), 0);
  for (int remaining = static_cast<int>(n); remaining > 1; --remaining) {
    std::swap(pi[remaining - 1], pi[g.draw_int(remaining)]);
  }
  return pi;
}

} // namespace

// data is flat; every words_per_slot consecutive words form one slot.
inline std::vector<word> permute(
    const std::vector<int>& pi,
    const std::vector<word>& data,
    std::size_t words_per_slot) {
  if (words_per_slot == 0) {
    throw std::invalid_argument("Words per slot must be positive");
  }
  if (data.size() % words_per_slot != 0 ||
      data.size() / words_per_slot != pi.size()) {
    throw std::invalid_argument("Permutation size must match data slot count");
  }
  std::vector<word> out(data.size());
  std::vector<bool> seen(pi.size(), false);
  for (std::size_t i = 0; i < pi.size(); ++i) {
    if (pi[i] < 0 || static_cast<std::size_t>(pi[i]) >= pi.size()) {
      throw std::invalid_argument("Permutation index out of range");
    }
    const auto destination = static_cast<std::size_t>(pi[i]);
    if (seen[destination]) {
      throw std::invalid_argument("Permutation contains duplicate index");
    }
    seen[destination] = true;
    for (std::size_t k = 0; k < words_per_slot; ++k) {
      out[destination * words_per_slot + k] = data[i * words_per_slot + k];
    }
  }
  return out;
}

struct p0_state {
  word seed;
  std::vector<int> pi0;
  std::vector<word> s;
  std::vector<word> r2;
};

struct p1_state {
  word seed;
  std::vector<int> pi1;
  std::vector<word> r;
  std::vector<word> t;
};

std::pair<p0_state, p1_state> preprocess_dealer(
    std::size_t slot_count, std::size_t slot_bits);

inline void prepare_p0(p0_state& state, std::size_t slot_count, std::size_t slot_bits) {
  if (slot_bits == 0) {
    throw std::invalid_argument("Shuffle slot width must be positive");
  }
  if (slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Shuffle slot count exceeds supported permutation range");
  }
  const std::size_t words_per_slot = slot_bits / word_bits + (slot_bits % word_bits != 0);
  if (slot_count > state.s.max_size() / words_per_slot) {
    throw std::length_error("Shuffle mask size exceeds supported vector size");
  }
  state.s.resize(slot_count * words_per_slot);

  prg g(state.seed);
  state.pi0 = sample_permutation(slot_count, g);
  const int last_word_bits = static_cast<int>((slot_bits - 1) % word_bits + 1);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const std::size_t begin = slot * words_per_slot;
    for (std::size_t k = 0; k < words_per_slot; ++k) {
      state.s[begin + k] = g();
    }
    state.s[begin + words_per_slot - 1] = state.s[begin + words_per_slot - 1].masked(last_word_bits);
  }
}

inline void prepare_p1(p1_state& state, std::size_t slot_count, std::size_t slot_bits) {
  if (slot_bits == 0) {
    throw std::invalid_argument("Shuffle slot width must be positive");
  }
  if (slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Shuffle slot count exceeds supported permutation range");
  }
  const std::size_t words_per_slot = slot_bits / word_bits + (slot_bits % word_bits != 0);
  if (slot_count > state.r.max_size() / words_per_slot) {
    throw std::length_error("Shuffle mask size exceeds supported vector size");
  }
  state.r.resize(slot_count * words_per_slot);
  state.t.resize(slot_count * words_per_slot);

  prg g(state.seed);
  state.pi1 = sample_permutation(slot_count, g);
  const int last_word_bits = static_cast<int>((slot_bits - 1) % word_bits + 1);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const std::size_t begin = slot * words_per_slot;
    for (std::size_t k = 0; k < words_per_slot; ++k) {
      state.r[begin + k] = g();
      state.t[begin + k] = g();
    }
    state.r[begin + words_per_slot - 1] = state.r[begin + words_per_slot - 1].masked(last_word_bits);
    state.t[begin + words_per_slot - 1] = state.t[begin + words_per_slot - 1].masked(last_word_bits);
  }
}

inline std::vector<word> shuffle_p0(
    const p0_state& state, const std::vector<word>& input_share, std::size_t slot_bits) {
  if (slot_bits == 0) {
    throw std::invalid_argument("Shuffle slot width must be positive");
  }
  if (party_id() != 0) {
    throw std::invalid_argument("shuffle_p0 requires party 0");
  }
  const std::size_t words = slot_bits / word_bits + (slot_bits % word_bits != 0);
  if (input_share.size() % words != 0 || input_share.size() / words != state.pi0.size()) {
    throw std::invalid_argument("Permutation size must match data slot count");
  }
  if (state.s.size() != input_share.size() || state.r2.size() != input_share.size()) {
    throw std::invalid_argument("P0 shuffle masks must match input size");
  }
  // Allocate all working storage before communication.
  std::vector<word> message(input_share.size());
  std::vector<word> output = state.r2;
  const int last_bits = static_cast<int>((slot_bits - 1) % word_bits + 1);
  for (std::size_t slot = 0; slot < state.pi0.size(); ++slot) {
    const auto destination = static_cast<std::size_t>(state.pi0[slot]);
    for (std::size_t k = 0; k < words; ++k) {
      message[destination * words + k] = input_share[slot * words + k] ^ recv_word(1);
    }
  }
  for (std::size_t i = 0; i < message.size(); ++i) {
    message[i] ^= state.s[i];
    if (i % words == words - 1) {
      message[i] = message[i].masked(last_bits);
      output[i] = output[i].masked(last_bits);
    }
    send_word(1, message[i]);
  }
  return output;
}

inline std::vector<word> shuffle_p1(
    const p1_state& state, const std::vector<word>& input_share, std::size_t slot_bits) {
  if (slot_bits == 0) {
    throw std::invalid_argument("Shuffle slot width must be positive");
  }
  if (party_id() != 1) {
    throw std::invalid_argument("shuffle_p1 requires party 1");
  }
  const std::size_t words = slot_bits / word_bits + (slot_bits % word_bits != 0);
  if (input_share.size() % words != 0 || input_share.size() / words != state.pi1.size()) {
    throw std::invalid_argument("Permutation size must match data slot count");
  }
  if (state.r.size() != input_share.size() || state.t.size() != input_share.size()) {
    throw std::invalid_argument("P1 shuffle masks must match input size");
  }
  std::vector<word> output(input_share.size());
  const int last_bits = static_cast<int>((slot_bits - 1) % word_bits + 1);
  for (std::size_t i = 0; i < input_share.size(); ++i) {
    word message = input_share[i] ^ state.r[i];
    if (i % words == words - 1) message = message.masked(last_bits);
    send_word(0, message);
  }
  for (std::size_t slot = 0; slot < state.pi1.size(); ++slot) {
    const auto destination = static_cast<std::size_t>(state.pi1[slot]);
    for (std::size_t k = 0; k < words; ++k) {
      word value = recv_word(0) ^ state.t[destination * words + k];
      if (k == words - 1) value = value.masked(last_bits);
      output[destination * words + k] = value;
    }
  }
  return output;
}

} // namespace dealer_shuffle

#endif
