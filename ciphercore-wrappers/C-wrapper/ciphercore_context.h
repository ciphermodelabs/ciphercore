#ifndef CIPHERCORE_CONTEXT_H
#define CIPHERCORE_CONTEXT_H

#include "ciphercore_structs.h"
#include "ciphercore_auxiliary_structs.h"
struct CResult_Context create_context(void);

struct CResult_Graph context_create_graph(struct Context *context_ptr);

struct CResult_Context context_finalize(struct Context *context_ptr);

struct CResult_Context context_set_main_graph(struct Context *context_ptr, struct Graph *graph_ptr);

struct CResult_CVec_Graph context_get_graphs(struct Context *context_ptr);

struct CResultVal_bool context_check_finalized(struct Context *context_ptr);

struct CResult_Graph context_get_main_graph(struct Context *context_ptr);

struct CResultVal_u64 context_get_num_graphs(struct Context *context_ptr);

struct CResult_Graph context_get_graph_by_id(struct Context *context_ptr, uint64_t id);

struct CResult_Node context_get_node_by_global_id(struct Context *context_ptr,
                                                  struct CVecVal_u64 global_id);

struct CResultVal_CStr context_to_string(struct Context *context_ptr);

struct CResultVal_bool contexts_deep_equal(struct Context *context1_ptr, struct Context *context2_ptr);

struct CResult_Context context_set_graph_name(struct Context *context_ptr,
                                              struct Graph *graph_ptr,
                                              struct CStr name);

struct CResultVal_CStr context_get_graph_name(struct Context *context_ptr, struct Graph *graph_ptr);

struct CResult_Graph context_retrieve_graph(struct Context *context_ptr, struct CStr name);

struct CResult_Context context_set_node_name(struct Context *context_ptr,
                                             struct Node *node_ptr,
                                             struct CStr name);

struct CResultVal_CStr context_get_node_name(struct Context *context_ptr, struct Node *node_ptr);

struct CResult_Node context_retrieve_node(struct Context *context_ptr,
                                          struct Graph *graph_ptr,
                                          struct CStr name);

void context_destroy(struct Context *context_ptr);
#endif //CIPHERCORE_CONTEXT_H

