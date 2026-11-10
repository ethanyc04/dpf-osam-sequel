#ifndef TYPES_H__
#define TYPES_H__

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

inline constexpr int word_bits = 128;
inline constexpr int u64_bits = 64;
inline constexpr int u32_bits = 32;
inline constexpr int word_u64_lanes = word_bits / u64_bits;
inline constexpr int word_u32_lanes = word_bits / u32_bits;

struct alignas(16) word {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;

  struct reference {
    word* owner = nullptr;
    std::size_t bit = 0;

    reference& operator=(bool value) noexcept {
      owner->set(bit, value);
      return *this;
    }

    reference& operator=(const reference& other) noexcept {
      return *this = static_cast<bool>(other);
    }

    reference& operator^=(bool value) noexcept {
      if (value) {
        owner->flip(bit);
      }
      return *this;
    }

    operator bool() const noexcept {
      return owner->test(bit);
    }
  };

  constexpr word() noexcept = default;
  constexpr word(unsigned long long value) noexcept
      : lo(static_cast<std::uint64_t>(value)), hi(0) { }

  static constexpr word from_u64s(std::uint64_t low, std::uint64_t high) noexcept {
    word out;
    out.lo = low;
    out.hi = high;
    return out;
  }

  constexpr bool test(std::size_t bit) const noexcept {
    if (bit < u64_bits) {
      return (lo >> bit) & 1ULL;
    }
    if (bit < word_bits) {
      return (hi >> (bit - u64_bits)) & 1ULL;
    }
    return false;
  }

  reference operator[](std::size_t bit) noexcept {
    return reference { this, bit };
  }

  constexpr bool operator[](std::size_t bit) const noexcept {
    return test(bit);
  }

  constexpr void set(std::size_t bit, bool value = true) noexcept {
    if (bit < u64_bits) {
      const std::uint64_t mask = 1ULL << bit;
      lo = value ? (lo | mask) : (lo & ~mask);
    } else if (bit < word_bits) {
      const std::uint64_t mask = 1ULL << (bit - u64_bits);
      hi = value ? (hi | mask) : (hi & ~mask);
    }
  }

  constexpr void reset(std::size_t bit) noexcept {
    set(bit, false);
  }

  constexpr void flip(std::size_t bit) noexcept {
    if (bit < u64_bits) {
      lo ^= 1ULL << bit;
    } else if (bit < word_bits) {
      hi ^= 1ULL << (bit - u64_bits);
    }
  }

  static constexpr std::uint64_t low_mask(int bits) noexcept {
    if (bits <= 0) {
      return 0;
    }
    if (bits >= u64_bits) {
      return ~std::uint64_t{0};
    }
    return (std::uint64_t{1} << bits) - 1;
  }

  constexpr word masked(int nbits) const noexcept {
    if (nbits <= 0) {
      return {};
    }
    if (nbits >= word_bits) {
      return *this;
    }
    if (nbits <= u64_bits) {
      return from_u64s(lo & low_mask(nbits), 0);
    }
    return from_u64s(lo, hi & low_mask(nbits - u64_bits));
  }

  constexpr void set_bits(int offset, int nbits, std::uint32_t value) noexcept {
    if (nbits <= 0) {
      return;
    }
    const std::uint64_t value64 = static_cast<std::uint64_t>(value);
    if (offset < u64_bits) {
      const int low_bits =
          nbits < (u64_bits - offset) ? nbits : (u64_bits - offset);
      const std::uint64_t mask = low_mask(low_bits);
      lo = (lo & ~(mask << offset)) | ((value64 & mask) << offset);
      const int high_bits = nbits - low_bits;
      if (high_bits > 0) {
        const std::uint64_t high_mask = low_mask(high_bits);
        hi = (hi & ~high_mask) | ((value64 >> low_bits) & high_mask);
      }
      return;
    }

    const int high_offset = offset - u64_bits;
    const std::uint64_t mask = low_mask(nbits);
    hi = (hi & ~(mask << high_offset)) | ((value64 & mask) << high_offset);
  }

  constexpr std::uint32_t get_bits(int offset, int nbits) const noexcept {
    if (nbits <= 0) {
      return 0;
    }
    if (offset < u64_bits) {
      const int low_bits =
          nbits < (u64_bits - offset) ? nbits : (u64_bits - offset);
      const std::uint64_t mask = low_mask(low_bits);
      std::uint64_t out = (lo >> offset) & mask;
      const int high_bits = nbits - low_bits;
      if (high_bits > 0) {
        out |= (hi & low_mask(high_bits)) << low_bits;
      }
      return static_cast<std::uint32_t>(out);
    }

    return static_cast<std::uint32_t>(
        (hi >> (offset - u64_bits)) & low_mask(nbits));
  }

  constexpr void set_u32_lane(int lane, std::uint32_t value) noexcept {
    set_bits(lane * u32_bits, u32_bits, value);
  }

  constexpr std::uint32_t u32_lane(int lane) const noexcept {
    return get_bits(lane * u32_bits, u32_bits);
  }

  unsigned long to_ulong() const {
    if (hi != 0 || lo > std::numeric_limits<unsigned long>::max()) {
      throw std::overflow_error("word does not fit in unsigned long");
    }
    return static_cast<unsigned long>(lo);
  }

  unsigned long long to_ullong() const {
    if (hi != 0) {
      throw std::overflow_error("word does not fit in unsigned long long");
    }
    return static_cast<unsigned long long>(lo);
  }

  explicit operator unsigned long() const {
    return to_ulong();
  }

  explicit operator unsigned long long() const {
    return to_ullong();
  }

  explicit operator std::uint32_t() const noexcept {
    return static_cast<std::uint32_t>(lo);
  }

  explicit operator int() const noexcept {
    return static_cast<int>(lo);
  }

  constexpr word& operator^=(word other) noexcept {
    lo ^= other.lo;
    hi ^= other.hi;
    return *this;
  }

  constexpr word& operator&=(word other) noexcept {
    lo &= other.lo;
    hi &= other.hi;
    return *this;
  }

  constexpr word& operator|=(word other) noexcept {
    lo |= other.lo;
    hi |= other.hi;
    return *this;
  }
};

static_assert(sizeof(word) == 16);
static_assert(alignof(word) == 16);

constexpr word operator^(word left, word right) noexcept {
  left ^= right;
  return left;
}

constexpr word operator&(word left, word right) noexcept {
  left &= right;
  return left;
}

constexpr word operator|(word left, word right) noexcept {
  left |= right;
  return left;
}

constexpr bool operator==(word left, word right) noexcept {
  return left.lo == right.lo && left.hi == right.hi;
}

constexpr bool operator!=(word left, word right) noexcept {
  return !(left == right);
}

#endif