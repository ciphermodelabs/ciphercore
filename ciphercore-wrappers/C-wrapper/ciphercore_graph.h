#ifndef CIPHERCORE_GRAPH_H
#define CIPHERCORE_GRAPH_H
#include "ciphercore_structs.h"
#include "ciphercore_auxiliary_structs.h"

struct CResult_Node graph_input(struct Graph *graph_ptr, struct Type *type_ptr);

struct CResult_Node graph_add(struct Graph *graph_ptr, struct Node *a_ptr, struct Node *b_ptr);

struct CResult_Node graph_subtract(struct Graph *graph_ptr, struct Node *a_ptr, struct Node *b_ptr);

struct CResult_Node graph_multiply(struct Graph *graph_ptr, struct Node *a_ptr, struct Node *b_ptr);

struct CResult_Node graph_mixed_multiply(struct Graph *graph_ptr, struct Node *a_ptr, struct Node *b_ptr);

struct CResult_Node graph_dot(struct Graph *graph_ptr, struct Node *a_ptr, struct Node *b_ptr);

struct CResult_Node graph_matmul(struct Graph *graph_ptr, struct Node *a_ptr, struct Node *b_ptr);

struct CResult_Node graph_cuckoo_hash(struct Graph *graph_ptr, struct Node *a_ptr, struct Node *b_ptr);

struct CResult_Node graph_truncate(struct Graph *graph_ptr, struct Node *a_ptr, uint64_t scale);

struct CResult_Node graph_sum(struct Graph *graph_ptr, struct Node *a_ptr, struct CVecVal_u64 axis);

struct CResult_Node graph_permute_axes(struct Graph *graph_ptr,
                                       struct Node *a_ptr,
                                       struct CVecVal_u64 axis);

struct CResult_Node graph_get(struct Graph *graph_ptr, struct Node *a_ptr, struct CVecVal_u64 index);

struct CResult_Node graph_get_slice(struct Graph *graph_ptr,
                                    struct Node *a_ptr,
                                    struct CSlice cslice);

struct CResult_Node graph_reshape(struct Graph *graph_ptr,
                                  struct Node *a_ptr,
                                  struct Type *new_type_ptr);

struct CResult_Node graph_random(struct Graph *graph_ptr, struct Type *output_type_ptr);

struct CResult_Node graph_stack(struct Graph *graph_ptr,
                                struct CVec_Node nodes,
                                struct CVecVal_u64 outer_shape);

struct CResult_Node graph_constant(struct Graph *graph_ptr, struct CTypedValue typed_value);

struct CResult_Node graph_a2b(struct Graph *graph_ptr, struct Node *a_ptr);

struct CResult_Node graph_b2a(struct Graph *graph_ptr,
                              struct Node *a_ptr,
                              struct ScalarType *scalar_type_ptr);

struct CResult_Node graph_create_tuple(struct Graph *graph_ptr, struct CVec_Node elements);

struct CResult_Node graph_create_vector(struct Graph *graph_ptr,
                                        struct Type *type_ptr,
                                        struct CVec_Node elements);

struct CResult_Node graph_create_named_tuple(struct Graph *graph_ptr,
                                             struct CVec_Node elements_nodes,
                                             struct CVecVal_CStr elements_names);

struct CResult_Node graph_tuple_get(struct Graph *graph_ptr,
                                    struct Node *tuple_node_ptr,
                                    uint64_t index);

struct CResult_Node graph_named_tuple_get(struct Graph *graph_ptr,
                                          struct Node *tuple_node_ptr,
                                          struct CStr key);

struct CResult_Node graph_vector_get(struct Graph *graph_ptr,
                                     struct Node *vec_node_ptr,
                                     struct Node *index_node_ptr);

struct CResult_Node graph_zip(struct Graph *graph_ptr, struct CVec_Node elements);

struct CResult_Node graph_repeat(struct Graph *graph_ptr, struct Node *a_ptr, uint64_t n);

struct CResult_Node graph_call(struct Graph *graph_ptr,
                               struct Graph *callee_ptr,
                               struct CVec_Node arguments);

struct CResult_Node graph_iterate(struct Graph *graph_ptr,
                                  struct Graph *callee_ptr,
                                  struct Node *state_ptr,
                                  struct Node *input_ptr);

struct CResult_Node graph_vector_to_array(struct Graph *graph_ptr, struct Node *a_ptr);

struct CResult_Node graph_array_to_vector(struct Graph *graph_ptr, struct Node *a_ptr);

struct CResult_Node graph_custom_op(struct Graph *graph_ptr,
                                    struct CCustomOperation c_custom_op,
                                    struct CVec_Node args);

struct CResult_Graph graph_finalize(struct Graph *graph_ptr);

struct CResult_CVec_Node graph_get_nodes(struct Graph *graph_ptr);

struct CResultVal_bool graph_set_output_node(struct Graph *graph_ptr, struct Node *n_ptr);

struct CResult_Node graph_get_output_node(struct Graph *graph_ptr);

struct CResultVal_u64 graph_get_id(struct Graph *graph_ptr);

struct CResultVal_u64 graph_get_num_nodes(struct Graph *graph_ptr);

struct CResult_Node graph_get_node_by_id(struct Graph *graph_ptr, uint64_t id);

struct CResult_Context graph_get_context(struct Graph *graph_ptr);

struct CResult_Graph graph_set_as_main(struct Graph *graph_ptr);

struct CResult_Graph graph_set_name(struct Graph *graph_ptr, struct CStr name);

struct CResultVal_CStr graph_get_name(struct Graph *graph_ptr);

struct CResult_Node graph_retrieve_node(struct Graph *graph_ptr, struct CStr name);

void graph_destroy(struct Graph *graph_ptr);
#endif //CIPHERCORE_GRAPH_H

