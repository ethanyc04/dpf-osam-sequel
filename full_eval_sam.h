#ifndef FULL_EVAL_SAM_H
#define FULL_EVAL_SAM_H

#include "dpf.h"
#include "preprocessing.h"
#include "util.h"

#include <array>
#include <vector>

namespace sam::full_eval {

using ::word;

// struct resource_plan {
//   int capacity = 0;
//   int write_budget = 0;
//   int value_bits = word_bits;
//   int allocs = 0;
//   int writes = 0;
// };

struct resources {
  // Planning resources have no DPF keys or permutation. They execute clear
  // memory semantics and grow on demand so a dry run can discover exact sizes.
  bool planning = false;
  std::vector<int> pi;
  std::vector<dpf_corr::correlation> write_corrs;
  int storage_size = 0;
  int value_bits = word_bits;
};

struct state {
  resources prep;
  std::vector<word> storage;
  std::size_t t_alloc;
  std::size_t t_write;
  std::size_t t_read;
};

inline void check_value_bits(int value_bits) {
  if (value_bits <= 0 || value_bits > word_bits) {
    throw std::invalid_argument("SAM value width out of range");
  }
}

// inline resources planning_resources(int storage_size, int value_bits = word_bits) {
//   if (storage_size < 0) {
//     throw std::invalid_argument("SAM planning capacity must be non-negative");
//   }
//   check_value_bits(value_bits);
//   return {
//     .planning = true,
//     .storage_size = storage_size,
//     .value_bits = value_bits,
//   };
// }

inline int logical_capacity(const resources& r) {
  return r.storage_size;
}

inline int logical_capacity(const state& s) {
  if (s.prep.planning) {
    return static_cast<int>(s.storage.size());
  }
  return logical_capacity(s.prep);
}

inline int logical_addr_bits(const state& s) {
  return lg(logical_capacity(s));
}

// Offline dealer: return P1's and P2's resources, indexed by party.
std::array<resources, 2> preprocess_dealer(
    int capacity, int max_writes, int value_bits = word_bits);

state initialize_from_resources(resources r);

int alloc(state& s);

// Both parties call with their own XOR shares; consumes one fresh correlation.
void write(state& s, int addr_share, word val_share);

word read_opened(state& s, int addr);

word read(state& s, int addr_share);

} // namespace sam::full_eval

#endif
