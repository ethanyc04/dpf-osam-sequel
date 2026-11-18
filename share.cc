#include "share.h"
#include "backend.h"

#include <cstdint>
#include <stdexcept>

namespace share {

int open(int x, int nbits) {
  if (nbits < 0 || nbits > u32_bits) {
    throw std::invalid_argument("Invalid int open width");
  }
  if (nbits == 0) return 0;
  const auto mask = nbits == u32_bits ? ~std::uint32_t{0} : (std::uint32_t{1} << nbits) - 1;
  const int own = static_cast<int>(static_cast<std::uint32_t>(x) & mask);
  int other;
  if (party_id() == 0) {
    other = recv_addr();
    send_addr(own);
  } else {
    send_addr(own);
    other = recv_addr();
  }
  return static_cast<int>(static_cast<std::uint32_t>(own ^ other) & mask);
}

word open(word x, int nbits) {
  if (nbits < 0 || nbits > word_bits) {
    throw std::invalid_argument("Invalid word open width");
  }
  if (nbits == 0) return {};
  const int own_party = party_id();
  const int peer = 1 - own_party;
  const word own = x.masked(nbits);
  word other;
  if (own_party == 0) {
    other = recv_word(peer);
    send_word(peer, own);
  } else {
    send_word(peer, own);
    other = recv_word(peer);
  }
  return (own ^ other).masked(nbits);
}

} // namespace share
