#ifndef SHARE_H
#define SHARE_H

#include "types.h"

namespace share {

// Open one XOR-shared value to both parties. Parties must call in matching order.
int open(int, int nbits = u32_bits);
word open(word, int nbits = word_bits);

} // namespace share

#endif
