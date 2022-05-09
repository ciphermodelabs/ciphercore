#include "stdio.h"
#include "ciphercore_adapters.h"

int test_wellformed_cases() {
   printf("running test: %s\n\r", __func__);
   CResult_Context context_res;
   CResult_Node node_res;
   CResult_Graph graph_res;
   CResult_Type type_res;
   CResultVal_bool bool_res;
   CResultVal_CStr cstr_res;

   context_res = create_context(); 
   Context * context = UNWRAP(context_res);

   graph_res = context_create_graph(context);
   Graph * graph = UNWRAP(graph_res);
   
   type_res = scalar_type(&BIT);
   Type *  t1 = UNWRAP(type_res);

   node_res = graph_input(graph, t1);
   Node * n1 = UNWRAP(node_res);
   node_res = graph_input(graph, t1);
   Node * n2 = UNWRAP(node_res);

   node_res = graph_add(graph, n1, n2);
   Node * n3 = UNWRAP(node_res);
   node_res = graph_subtract(graph, n1, n2);
   Node * n4 = UNWRAP(node_res);
   node_res = graph_multiply(graph, n1, n2);
   Node * n5 = UNWRAP(node_res);
   node_res = graph_dot(graph, n1, n2);
   Node * n6 = UNWRAP(node_res);
//   node_res = graph_matmul(graph, n1, n2);// will panic because of type checker
//   UNWRAP(node_res);
   node_res = graph_truncate(graph, n1, 123);
   Node * n7 = UNWRAP(node_res);

   uint64_t arr[3] = {10,20,30}; 
   CVecVal_u64 array_shape = {((uint64_t*)arr),3};
   type_res = array_type(array_shape, &BIT);
   Type * t2 = UNWRAP(type_res);
   node_res = graph_input(graph,t2);
   Node * n8 = UNWRAP(node_res);

   uint64_t arr2[2] = {1,2}; 
   CVecVal_u64 array_shape2 = {((uint64_t*)arr2),2};
   node_res = graph_sum(graph,n8,array_shape2);
   Node *n9 = UNWRAP(node_res);

   uint64_t arr3[3] = {1,2,0}; 
   CVecVal_u64 array_shape3 = {((uint64_t*)arr3),3};
   node_res = graph_permute_axes(graph,n8,array_shape3);
   Node * permuted = UNWRAP(node_res);

   node_res = graph_get(graph,n8,array_shape2);
   Node * n10 = UNWRAP(node_res);

   uint64_t arr4[2] = {20,300}; 
   CVecVal_u64 array_shape4 = {((uint64_t*)arr4),2};
   type_res = array_type(array_shape4,&BIT);
   Type * t3 = UNWRAP(type_res);
   node_res = graph_reshape(graph,n8,t3);
   Node * n11 = UNWRAP(node_res);
   

 //  UNWRAP(graph_nop(n8)); 
 //Not implemented because it is not part of globaly visible functions (API)

   uint64_t arr5[1] = {128}; 
   CVecVal_u64 array_shape5 = {((uint64_t*)arr5),1};
   type_res = array_type(array_shape5,&BIT);
   Type * t4 = UNWRAP(type_res);
   node_res = graph_random(graph,t4);
   Node * key = UNWRAP(node_res);


   //uint64_t arr6[2] = {10,10}; 
   //CVecVal_u64 array_shape6 = {((uint64_t*)arr6),2};
   //Type * t5 = UNWRAP(array_type(array_shape6,&UINT64));
   //UNWRAP(graph_prf(graph,key,0,t5));  
   // NOT implemented because it is not part of globaly visible functions (API)

   uint64_t arr7[2] = {2,1}; 
   CVecVal_u64 array_shape7 = {((uint64_t*)arr7),2};
   Node * node_arr[2] = {n1,n2};
   CVec_Node nodes = {(Node **)node_arr,2};
   node_res = graph_stack(graph,nodes,array_shape7);
   Node * n12 = UNWRAP(node_res);

   char json[256] = "{\"kind\":\"scalar\",\"type\":\"u8\",\"value\":1}";
   CStr cstr_json = {(uint8_t*) json};
   CTypedValue tv = {cstr_json};
   node_res = graph_constant(graph,tv);
   Node * c = UNWRAP(node_res);

   uint64_t arr8[2] = {10,10}; 
   CVecVal_u64 array_shape8 = {((uint64_t*)arr8),2};
   type_res = array_type(array_shape8,&UINT64);
   Type * t5 = UNWRAP(type_res);
   node_res = graph_input(graph,t5);
   Node * n13 = UNWRAP(node_res);

   node_res = graph_a2b(graph,n13);
   Node * bits = UNWRAP(node_res);

   node_res = graph_b2a(graph,bits,(struct ScalarType *)&UINT64);
   Node * n14 = UNWRAP(node_res);

   node_res = graph_create_tuple(graph, nodes);
   Node * tup = UNWRAP(node_res);

   node_res = graph_create_vector(graph,t1,nodes);
   Node * vec = UNWRAP(node_res); 

   CVecVal_CStr names;
   CStr name1 = {"Name"};
   CStr name2 = {"Gender"};
   CStr names_arr[2] = {name1,name2};
   names.ptr = names_arr;
   names.len = 2;
   node_res = graph_create_named_tuple(graph,nodes,names);
   Node * named_tup = UNWRAP(node_res);

   node_res = graph_tuple_get(graph,tup,1);
   Node * tup_elem = UNWRAP(node_res);   

   CStr q_name = {"Gender"};
   node_res = graph_named_tuple_get(graph,named_tup,q_name);   
   Node * n15 = UNWRAP(node_res);   

   node_res = graph_repeat(graph,c,100);
   Node * n16 = UNWRAP(node_res);   

   Node * node_arr2[4] = {vec,vec,vec,vec};
   CVec_Node nodes2 = {(Node **)node_arr2,4};
   node_res = graph_zip(graph,nodes2);
   Node * n17 = UNWRAP(node_res);   

   char json2[256] = "{\"kind\":\"scalar\",\"type\":\"u64\",\"value\":0}";
   CStr cstr_json2 = {(uint8_t*) json2};
   CTypedValue tv2 = {cstr_json2};
   node_res = graph_constant(graph,tv2);
   Node * zero = UNWRAP(node_res);

   node_res = graph_vector_get(graph,vec,zero);
   Node * n18 = UNWRAP(node_res);   

   node_res = graph_vector_to_array(graph,vec);
   Node * array = UNWRAP(node_res);

   node_res = graph_array_to_vector(graph, array);
   Node * vec2 = UNWRAP(node_res);

   bool_res = graph_set_output_node(graph,vec2);
   UNWRAP(bool_res);

   graph_res = graph_finalize(graph);
   Graph * graph2 = UNWRAP(graph_res);

   context_res = context_set_main_graph(context,graph);
   Context * context2 = UNWRAP(context_res);

   context_res = context_finalize(context);
   Context * context3 = UNWRAP(context_res);

   cstr_res = context_to_string(context);
   CStr context_str = UNWRAP(cstr_res);
   printf("%s\n\r",context_str.ptr);

   cstr_destroy(context_str);

   CResult_CVecVal_u64 g_id_res = node_get_global_id(n18);
   CVecVal_u64 * g_id = UNWRAP(g_id_res);   
   printf("g id = %ld, %ld, len=%ld\n\r",g_id->ptr[0] , g_id->ptr[1] , g_id->len);

   cvec_u64_destroy(g_id);

   CResult_CVec_Node all_nodes_res = graph_get_nodes(graph);
   CVec_Node * all_nodes = UNWRAP(all_nodes_res);
   printf("nodes len=%ld\n\r" , all_nodes->len);
   Node * nx5 = all_nodes->ptr[5];
   CResultVal_u64 id_res = node_get_id(nx5);
   uint64_t nxid5 = UNWRAP(id_res);
   printf("node x5 id: %ld\n\r", nxid5);

   uint64_t len = all_nodes->len;
   for (uint64_t i =0 ; i<len ; i++){
      node_destroy(all_nodes->ptr[i]);
   }

   cvec_node_destroy(all_nodes);
   

   type_destroy(t1);
   type_destroy(t2);
   type_destroy(t3);
   type_destroy(t4);
   type_destroy(t5);

   node_destroy(n1);
   node_destroy(n2);
   node_destroy(n3);
   node_destroy(n4);
   node_destroy(n5);
   node_destroy(n6);
   node_destroy(n7);
   node_destroy(n8);
   node_destroy(n9);
   node_destroy(n10);
   node_destroy(n11);
   node_destroy(n12);
   node_destroy(n13);
   node_destroy(n14);
   node_destroy(n15);
   node_destroy(n16);
   node_destroy(n17);
   node_destroy(n18);
   node_destroy(permuted);
   node_destroy(key);
   node_destroy(c);
   node_destroy(bits);
   node_destroy(tup);
   node_destroy(vec);
   node_destroy(named_tup);
   node_destroy(tup_elem);
   node_destroy(zero);
   node_destroy(array);
   node_destroy(vec2);

  
   graph_destroy(graph);
   graph_destroy(graph2);
   context_destroy(context2);
   context_destroy(context3);
   context_destroy(context);
   printf("\t success!\n\r");
   return 0;
}
int main() {

   return test_wellformed_cases();
}

