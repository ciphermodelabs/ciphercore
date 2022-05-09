#ifndef CSTR_H
#define CSTR_H
#include <stdint.h>

typedef struct CStr {
  const uint8_t *ptr;
} CStr;

struct CResultVal_bool cstr_destroy(struct CStr cstr);

#endif //CSTR_H

