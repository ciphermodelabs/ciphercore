#ifndef CIPHERCORE_NODE_H
#define CIPHERCORE_NODE_H

#include "ciphercore_structs.h"
#include "ciphercore_auxiliary_structs.h"
struct CResult_Graph node_get_graph(struct Node *node_ptr);

struct CResult_CVec_Node node_get_dependencies(struct Node *node_ptr);

struct CResult_CVec_Graph node_graph_dependencies(struct Node *node_ptr);

struct CResult_COperation node_get_operation(struct Node *node_ptr);

struct CResultVal_u64 node_get_id(struct Node *node_ptr);

struct CResult_CVecVal_u64 node_get_global_id(struct Node *node_ptr);

struct CResult_Type node_get_type(struct Node *node_ptr);

struct CResult_Node node_add(struct Node *node_ptr, struct Node *b_ptr);

struct CResult_Node node_subtract(struct Node *node_ptr, struct Node *b_ptr);

struct CResult_Node node_multiply(struct Node *node_ptr, struct Node *b_ptr);

struct CResult_Node node_mixed_multiply(struct Node *node_ptr, struct Node *b_ptr);

struct CResult_Node node_dot(struct Node *node_ptr, struct Node *b_ptr);

struct CResult_Node node_matmul(struct Node *node_ptr, struct Node *b_ptr);

struct CResult_Node node_truncate(struct Node *node_ptr, uint64_t scale);

struct CResult_Node node_sum(struct Node *node_ptr, struct CVecVal_u64 axis);

struct CResult_Node node_permute_axes(struct Node *node_ptr, struct CVecVal_u64 axis);

struct CResult_Node node_get(struct Node *node_ptr, struct CVecVal_u64 index);

struct CResult_Node node_get_slice(struct Node *node_ptr, struct CSlice cslice);

struct CResult_Node node_reshape(struct Node *node_ptr, struct Type *type_ptr);

struct CResult_Node node_nop(struct Node *node_ptr);

struct CResult_Node node_prf(struct Node *node_ptr, uint64_t iv, struct Type *output_type_ptr);

struct CResult_Node node_a2b(struct Node *node_ptr);

struct CResult_Node node_b2a(struct Node *node_ptr, struct ScalarType *scalar_type_ptr);

struct CResult_Node node_tuple_get(struct Node *node_ptr, uint64_t index);

struct CResult_Node node_named_tuple_get(struct Node *node_ptr, struct CStr key);

struct CResult_Node node_vector_get(struct Node *node_ptr, struct Node *index_node_ptr);

struct CResult_Node node_array_to_vector(struct Node *node_ptr);

struct CResult_Node node_vector_to_array(struct Node *node_ptr);

struct CResult_Node node_repeat(struct Node *node_ptr, uint64_t n);

struct CResult_Node node_set_as_output(struct Node *node_ptr);

void node_destroy(struct Node *node_ptr);
#endif //CIPHERCORE_NODE_H

