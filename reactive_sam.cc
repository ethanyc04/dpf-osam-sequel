#include "reactive_sam.h"
#include "share.h"

#include <stdexcept>
#include <utility>

namespace sam::reactive {

state initialize_from_resources(resources r) {
  if (!is_power_of_two(r.storage_size)) {
    throw std::invalid_argument("SAM capacity must be a positive power of two");
  }
  check_value_bits(r.value_bits);
  const auto capacity = static_cast<std::size_t>(r.storage_size);
  if (r.pi.size() != capacity) {
    throw std::invalid_argument("SAM address space must match capacity");
  }

  return {
    .prep = std::move(r),
    .writes = {},
    .t_alloc = 0,
    .t_write = 0,
    .t_read = 0,
    .t_dpf_eval = 0,
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
  dpf::detail::validate_address(addr_share, logical_capacity(s));
  s.writes.push_back(std::make_tuple(addr_share, val_share));
  ++s.t_write;
}

word read_opened(state& s, int addr) {
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
  if (s.t_dpf_eval > s.prep.dpf_corrs.size() ||
      s.writes.size() > s.prep.dpf_corrs.size() - s.t_dpf_eval) {
    throw std::runtime_error("Insufficient DPF correlations");
  }
  check_value_bits(s.prep.value_bits);
  word val_share;
  for (std::size_t i = 0; i < s.writes.size(); ++i) {
    const auto& [write_addr_share, write_val_share] = s.writes[i];
    const int addr_bits = lg(s.prep.storage_size);
    const auto& corr = s.prep.dpf_corrs[s.t_dpf_eval++];
    if (corr.key.size() != static_cast<std::size_t>(addr_bits + 1)) {
      throw std::invalid_argument("DPF correlation domain must match capacity");
    }
    const int offset = share::open(write_addr_share ^ corr.addr_share, addr_bits);
    const word masked_value = share::open(corr.key.mask_share ^ write_val_share, s.prep.value_bits);
    const word leaf = dpf::eval(corr.key, addr ^ offset);
    val_share ^= leaf ^ (leaf[0] ? masked_value : word{});
  }
  val_share = val_share.masked(s.prep.value_bits);
  ++s.t_read;
  return val_share;
}

word read(state& s, int addr_share) {
  const auto addr = share::open(addr_share, logical_addr_bits(s));
  return read_opened(s, addr);
}

} // namespace sam::reactive
