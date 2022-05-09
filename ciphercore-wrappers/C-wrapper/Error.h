#ifndef ERROR_H
#define ERROR_H
#include "CStr.h"

#define PANIC(x) {fprintf(stderr,"panic: %s\n\r",x.msg.ptr); exit(-1);}

typedef enum CiphercoreErrorKind {
  RuntimeError,
} CiphercoreErrorKind;

typedef struct CiphercoreError {
  enum CiphercoreErrorKind kind;
  struct CStr msg;
} CiphercoreError;

#endif //ERROR_H

