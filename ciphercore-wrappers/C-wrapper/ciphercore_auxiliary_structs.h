#ifndef CIPHERCORE_AUXILIARY_STRUCTS_H
#define CIPHERCORE_AUXILIARY_STRUCTS_H

#include <stdint.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "CVec.h"
#include "CResults.h"
#include "CStr.h"

typedef struct COption_i64 {
  bool valid;
  int64_t num;
} COption_i64;

typedef struct COption_i64_triplet {
  struct COption_i64 op1;
  struct COption_i64 op2;
  struct COption_i64 op3;
} COption_i64_triplet;

typedef enum CSliceElement_Tag {
  SingleIndex,
  SubArray,
  Ellipsis,
} CSliceElement_Tag;

typedef struct CSliceElement {
  CSliceElement_Tag tag;
  union {
    struct {
      int64_t single_index;
    };
    struct {
      struct COption_i64_triplet sub_array;
    };
  };
} CSliceElement;

typedef struct CSlice {
  struct CVec_CSliceElement elements;
} CSlice;

void c_slice_destroy(struct CSlice *cslice_ptr);


typedef struct CTypedValue {
  struct CStr json;
} CTypedValue;

typedef struct CCustomOperation {
  struct CStr json;
} CCustomOperation;

typedef struct U64TypePtrTuple {
  uint64_t iv;
  struct Type *type_ptr;
} U64TypePtrTuple;


typedef enum COperation_Tag {
  Input,
  Add,
  Subtract,
  Multiply,
  Dot,
  Matmul,
  Truncate,
  Sum,
  PermuteAxes,
  Get,
  GetSlice,
  Reshape,
  NOP,
  Random,
  PRF,
  Stack,
  Constant,
  A2B,
  B2A,
  CreateTuple,
  CreateNamedTuple,
  CreateVector,
  TupleGet,
  NamedTupleGet,
  VectorGet,
  Zip,
  Repeat,
  Call,
  Iterate,
  ArrayToVector,
  VectorToArray,
  Custom,
} COperation_Tag;

typedef struct COperation {
  COperation_Tag tag;
  union {
    struct {
      struct Type *input;
    };
    struct {
      uint64_t truncate;
    };
    struct {
      struct CVecVal_u64 *sum;
    };
    struct {
      struct CVecVal_u64 *permute_axes;
    };
    struct {
      struct CVecVal_u64 *get;
    };
    struct {
      struct CSlice *get_slice;
    };
    struct {
      struct Type *reshape;
    };
    struct {
      struct Type *random;
    };
    struct {
      struct U64TypePtrTuple *prf;
    };
    struct {
      struct CVecVal_u64 *stack;
    };
    struct {
      struct CTypedValue *constant;
    };
    struct {
      struct ScalarType *b2a;
    };
    struct {
      struct CVecVal_CStr *create_named_tuple;
    };
    struct {
      struct Type *create_vector;
    };
    struct {
      uint64_t tuple_get;
    };
    struct {
      struct CStr named_tuple_get;
    };
    struct {
      uint64_t repeat;
    };
    struct {
      struct CCustomOperation custom;
    };
  };
} COperation;

void c_operation_destroy(struct COperation *cop_ptr);

#endif //CIPHERCORE_AUXILIARY_STRUCTS_H

