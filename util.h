#ifndef UTIL_H__
#define UTIL_H__

#include "types.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <random>
#include <utility>

#ifdef __x86_64__
#include <immintrin.h>
#elif __aarch64__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "third_party/sse2neon.h"
#pragma GCC diagnostic pop
#endif

constexpr int lg(int x) {
  --x;
  int lgx = 0;
  while (x >= 1) {
    ++lgx;
    x >>= 1;
  }
  return lgx;
}

constexpr bool is_power_of_two(int x) {
  return x > 0 && (x & (x - 1)) == 0;
}

constexpr int ceil_power_of_two(int x) {
  if (x <= 1) {
    return 1;
  }
  const std::uint64_t rounded = std::uint64_t{1} << lg(x);
  if (rounded > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("ceil_power_of_two result exceeds int range");
  }
  return static_cast<int>(rounded);
}

struct prf {
  inline static std::atomic_size_t num_calls = 0;
  static constexpr std::size_t nrounds = 10;

  prf();
  explicit prf(word);
  word eval(word inp) const;
  std::array<word, 2> eval_pair(word left, word right) const;
  word operator()(word inp) const;
  static void reset_calls();
  static void add_calls(std::size_t calls = 1);
  static std::size_t calls();

private:
  void load_round_keys();

  std::array<word, nrounds + 1> key;
  std::array<__m128i, nrounds + 1> round_key;
};

inline word rand_key() {
  std::random_device dev;
  const auto draw_u64 = [&] {
    return (static_cast<std::uint64_t>(dev()) << u32_bits)
      | static_cast<std::uint64_t>(dev());
  };
  return word::from_u64s(draw_u64(), draw_u64());
}

inline __m128i word_to_m128(word w) {
  return _mm_set_epi64x(
      static_cast<long long>(w.hi),
      static_cast<long long>(w.lo));
}

inline word word_from_m128(__m128i x) {
  alignas(16) std::uint64_t parts[word_u64_lanes];
  _mm_store_si128(reinterpret_cast<__m128i*>(parts), x);
  return word::from_u64s(parts[0], parts[1]);
}

template <int aes_const>
word expand_assist(word kk) {
  __m128i k = word_to_m128(kk);
  auto keygened = _mm_aeskeygenassist_si128(k, aes_const);
  keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3,3,3,3));
  k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
  k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
  k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
  k = _mm_xor_si128(k, keygened);
  return word_from_m128(k);
}

template <int round>
void expand(std::array<word, prf::nrounds + 1>& key) {
  if constexpr (round <= prf::nrounds) {
    key[round] = expand_assist<(1 << (round - 1)) % 229>(key[round - 1]);
    expand<round + 1>(key);
  }
}

inline prf::prf() {
  key[0] = rand_key();
  expand<1>(key);
  load_round_keys();
}

inline prf::prf(word packed) {
  key[0] = packed;
  expand<1>(key);
  load_round_keys();
}

inline void prf::load_round_keys() {
  for (std::size_t i = 0; i <= nrounds; ++i) {
    round_key[i] = word_to_m128(key[i]);
  }
}

inline word prf::eval(word inp) const {
  __m128i tar = word_to_m128(inp);
  tar = _mm_xor_si128(tar, round_key[0]);
  for (std::size_t i = 1; i < nrounds; ++i) {
    tar = _mm_aesenc_si128(tar, round_key[i]);
  }
  const auto out = _mm_aesenclast_si128(tar, round_key[nrounds]);
  return word_from_m128(out);
}

inline std::array<word, 2> prf::eval_pair(word left, word right) const {
  __m128i left_block = _mm_xor_si128(word_to_m128(left), round_key[0]);
  __m128i right_block = _mm_xor_si128(word_to_m128(right), round_key[0]);
  for (std::size_t i = 1; i < nrounds; ++i) {
    left_block = _mm_aesenc_si128(left_block, round_key[i]);
    right_block = _mm_aesenc_si128(right_block, round_key[i]);
  }
  left_block = _mm_aesenclast_si128(left_block, round_key[nrounds]);
  right_block = _mm_aesenclast_si128(right_block, round_key[nrounds]);
  return {
    word_from_m128(left_block),
    word_from_m128(right_block),
  };
}

inline word prf::operator()(word inp) const {
  add_calls();
  return eval(inp);
}

inline void prf::reset_calls() {
  num_calls.store(0, std::memory_order_relaxed);
}

inline void prf::add_calls(std::size_t calls) {
  num_calls.fetch_add(calls, std::memory_order_relaxed);
}

inline std::size_t prf::calls() {
  return num_calls.load(std::memory_order_relaxed);
}

struct prg {
public:
  prg() = default;
  explicit prg(prf f) : f(std::move(f)) { }
  explicit prg(word seed) : f(std::move(seed)) { }

  word operator()() {
    if (bit_pos == u32_bits && lane_pos == word_u32_lanes) {
      return f(nonce++);
    }

    word out;
    for (int lane = 0; lane < word_u32_lanes; ++lane) {
      out.set_u32_lane(lane, draw_u32());
    }
    return out;
  }

  bool bit() {
    return next_bit();
  }

  std::uint32_t draw_u32() {
    if (bit_pos == u32_bits) {
      if (lane_pos == word_u32_lanes) {
        refill();
      }
      return lanes[lane_pos++];
    }

    std::uint32_t out = 0;
    for (int bit = 0; bit < u32_bits; ++bit) {
      if (next_bit()) {
        out |= 1U << bit;
      }
    }
    return out;
  }

  int draw_int(int mod) {
    const std::uint32_t bound = static_cast<std::uint32_t>(mod);
    std::uint32_t mask = bound - 1;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;

    while (true) {
      const std::uint32_t out = draw_u32() & mask;
      if (out < bound) {
        return static_cast<int>(out);
      }
    }
  }

private:
  void refill() {
    const word w = f(nonce++);
    for (int lane = 0; lane < word_u32_lanes; ++lane) {
      lanes[lane] = w.u32_lane(lane);
    }
    lane_pos = 0;
    bit_pos = u32_bits;
  }

  bool next_bit() {
    if (bit_pos == u32_bits) {
      if (lane_pos == word_u32_lanes) {
        refill();
      }
      bit_lane = lanes[lane_pos++];
      bit_pos = 0;
    }
    return (bit_lane >> bit_pos++) & 1U;
  }

  prf f;
  std::size_t nonce = 0;
  std::array<std::uint32_t, word_u32_lanes> lanes {};
  int lane_pos = word_u32_lanes;
  std::uint32_t bit_lane = 0;
  int bit_pos = u32_bits;
};

#endif
