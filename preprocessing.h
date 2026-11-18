#ifndef PREPROCESSING_H__
#define PREPROCESSING_H__

#include "dpf.h"

#include <array>
#include <cstddef>

namespace dpf_corr {

// One party's preprocessing for a random-address, random-valued DPF.
struct correlation {
  int addr_share = 0; // The two shares XOR to the DPF's random address.
  dpf::key key;
};

// Offline dealer: return P1's and P2's correlations for a domain of size 2^lgn.
// Supports 0 <= lgn <= numeric_limits<int>::digits; otherwise throws invalid_argument.
std::array<correlation, 2> preprocess_dealer(std::size_t lgn);

} // namespace dpf_corr

#endif
