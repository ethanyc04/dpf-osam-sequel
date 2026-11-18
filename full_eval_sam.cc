#include "full_eval_sam.h"
#include "share.h"

#include <stdexcept>
#include <utility>

namespace sam::full_eval {

state initialize_from_resources(resources r) {
  if (!is_power_of_two(r.storage_size)) {
    throw std::invalid_argument("SAM capacity must be a positive power of two");
  }
  check_value_bits(r.value_bits);
  const auto capacity = static_cast<std::size_t>(r.storage_size);
  if (r.pi.size() != capacity) {
    throw std::invalid_argument("SAM allocation share count must match capacity");
  }

  return {
    .prep = std::move(r),
    .storage = std::vector<word>(capacity),
    .t_alloc = 0,
    .t_write = 0,
    .t_read = 0,
  };
}

int alloc(state& s) {
  // if (s.prep.planning) {
  //   const auto addr = static_cast<int>(s.t_alloc++);
  //   if (addr >= static_cast<int>(s.storage.size())) {
  //     s.storage.resize(static_cast<std::size_t>(addr + 1));
  //   }
  //   s.max_touched = std::max(s.max_touched, static_cast<std::size_t>(addr + 1));
  //   return addr;
  // }
  if (s.t_alloc >= s.prep.pi.size()) {
    throw std::runtime_error("SAM allocation budget exhausted");
  }
  const int addr = s.prep.pi[s.t_alloc++];
  if (addr < 0 || addr >= logical_capacity(s)) {
    throw std::runtime_error("SAM allocation permutation produced invalid address");
  }
  return addr;
}

void write(state& s, int addr_share, word val_share) {
  if (s.t_write >= s.prep.write_corrs.size()) {
    throw std::runtime_error("SAM write correlation budget exhausted");
  }
  if (s.prep.planning) {
    throw std::invalid_argument("SAM planning writes are not implemented");
  }
  if (!is_power_of_two(s.prep.storage_size)
      || s.storage.size() != static_cast<std::size_t>(s.prep.storage_size)) {
    throw std::invalid_argument("SAM storage size must match its power-of-two capacity");
  }
  check_value_bits(s.prep.value_bits);
  const int addr_bits = lg(s.prep.storage_size);
  const auto& corr = s.prep.write_corrs[s.t_write];
  if (corr.key.size() != static_cast<std::size_t>(addr_bits + 1)) {
    throw std::invalid_argument("SAM write correlation domain must match capacity");
  }
  dpf::detail::validate_address(addr_share, s.storage.size());
  dpf::detail::validate_address(corr.addr_share, s.storage.size());

  val_share = val_share.masked(s.prep.value_bits);
  const int offset = share::open(addr_share ^ corr.addr_share, addr_bits);
  const word masked_value = share::open(corr.key.mask_share ^ val_share, s.prep.value_bits);
  dpf::full_shift_mask_into(corr.key, offset, masked_value, s.storage);
  ++s.t_write;
}

word read_opened(state& s, int addr) {
  ++s.t_read;
  if (addr < 0) {
    throw std::runtime_error("SAM read to negative address");
  }
  // if (s.prep.planning && addr >= static_cast<int>(s.storage.size())) {
  //   s.storage.resize(static_cast<std::size_t>(addr + 1));
  // }
  // if (s.prep.planning) {
  //   s.max_touched = std::max(s.max_touched, static_cast<std::size_t>(addr + 1));
  //   return word_pack::mask(s.storage[addr], s.prep.value_bits);
  // }
  if (addr >= logical_capacity(s)) {
    throw std::runtime_error("SAM read past logical capacity");
  }
  return s.storage[addr].masked(s.prep.value_bits);
}

word read(state& s, int addr_share) {
  const auto addr = share::open(addr_share, logical_addr_bits(s));
  return read_opened(s, addr);
}
} // namespace sam::full_eval
