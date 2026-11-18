#ifndef REACTIVE_SAM_H
#define REACTIVE_SAM_H

#include "dpf.h"
#include "preprocessing.h"
#include "util.h"

#include <array>
#include <tuple>
#include <vector>

namespace sam::reactive {

using ::word;

struct resources {
  // Planning resources have no DPF keys or permutation. They execute clear
  // memory semantics and grow on demand so a dry run can discover exact sizes.
  bool planning = false;
  std::vector<int> pi;
  std::vector<dpf_corr::correlation> dpf_corrs;
  int storage_size = 0;
  int value_bits = word_bits;
};

struct state {
  resources prep;
  std::vector<std::tuple<int, word>> writes;
  std::size_t t_alloc;
  std::size_t t_write;
  std::size_t t_read;
  std::size_t t_dpf_eval;
};

inline void check_value_bits(int value_bits) {
  if (value_bits <= 0 || value_bits > word_bits) {
    throw std::invalid_argument("SAM value width out of range");
  }
}

inline int logical_capacity(const resources& r) {
  return r.storage_size;
}

inline int logical_capacity(const state& s) {
  return logical_capacity(s.prep);
}

inline int logical_addr_bits(const state& s) {
  return lg(logical_capacity(s));
}

// Offline dealer: return P1's and P2's resources, indexed by party.
std::array<resources, 2> preprocess_dealer(
    int capacity, int max_dpf_corrs, int value_bits = word_bits);

state initialize_from_resources(resources r);

int alloc(state& s);

void write(state& s, int addr_share, word val_share);

word read_opened(state& s, int addr);

word read(state& s, int addr_share);

} // namespace sam::reactive

#endif
