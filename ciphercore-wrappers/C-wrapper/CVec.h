#ifndef CVEC_H
#define CVEC_H

#include <stdint.h>
#include "ciphercore_auxiliary_structs.h"
#include "CStr.h"


typedef uint64_t u64;

#define CVEC(T) \
typedef struct CVec_##T {\
  struct T **ptr; \
  uintptr_t len; \
} CVec_##T \

#define CVEC_VAL(T) \
typedef struct CVecVal_##T {\
  T *ptr; \
  uintptr_t len; \
} CVecVal_##T \


CVEC(Type);
struct CResultVal_bool cvec_type_destroy(struct CVec_Type *cvec_ptr);


CVEC(CSliceElement);
struct CResultVal_bool cvec_cslice_element_destroy(struct CVec_CSliceElement *cvec_ptr);

CVEC(Node);
struct CResultVal_bool cvec_node_destroy(struct CVec_Node *cvec_ptr);

CVEC(Graph);
struct CResultVal_bool cvec_graph_destroy(struct CVec_Graph *cvec_ptr);

CVEC_VAL(u64);
struct CResultVal_bool cvec_u64_destroy(struct CVecVal_u64 *cvec_ptr);


CVEC_VAL(CStr);
struct CResultVal_bool cvec_cstr_destroy(struct CVecVal_CStr *cvec_ptr);
#endif //CVEC_H

