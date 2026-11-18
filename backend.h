#ifndef BACKEND_H
#define BACKEND_H

#include "types.h"

// Each online party runs in its own thread and explicitly selects ID 0 or 1.
void set_party_id(int);
int party_id();
const char* backend_name();
void send_addr(int x);
int recv_addr();
void send_word(int dst_party, word x);
word recv_word(int src_party);

#endif
