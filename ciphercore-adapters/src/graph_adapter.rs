extern crate ciphercore_base;

use crate::adapters_utils::{
    destroy_helper, unsafe_deref, CCustomOperation, COperation, CResult, CResultTrait, CResultVal,
    CSlice, CStr, CTypedValue, CVec, CVecVal,
};
use ciphercore_base::data_types::{ScalarType, Type};
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::{Context, Graph, Node};
use ciphercore_base::runtime_error;

fn graph_method_helper<T, F, R: CResultTrait<T>>(graph_ptr: *mut Graph, op: F) -> R
where
    F: FnOnce(Graph) -> Result<T>,
{
    let helper = || -> Result<T> {
        let graph = unsafe_deref(graph_ptr)?;
        op(graph)
    };
    R::new(helper())
}

fn graph_one_node_method_helper<T, F, R: CResultTrait<T>>(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    op: F,
) -> R
where
    F: FnOnce(Graph, Node) -> Result<T>,
{
    let helper = || -> Result<T> {
        let graph = unsafe_deref(graph_ptr)?;
        let a = unsafe_deref(a_ptr)?;
        op(graph, a)
    };
    R::new(helper())
}

fn graph_two_nodes_method_helper<T, F, R: CResultTrait<T>>(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    b_ptr: *mut Node,
    op: F,
) -> R
where
    F: FnOnce(Graph, Node, Node) -> Result<T>,
{
    let helper = || -> Result<T> {
        let graph = unsafe_deref(graph_ptr)?;
        let a = unsafe_deref(a_ptr)?;
        let b = unsafe_deref(b_ptr)?;
        op(graph, a, b)
    };
    R::new(helper())
}

#[no_mangle]
pub extern "C" fn graph_input(graph_ptr: *mut Graph, type_ptr: *mut Type) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let t = unsafe_deref(type_ptr)?;
        g.input(t)
    })
}

#[no_mangle]
pub extern "C" fn graph_add(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    b_ptr: *mut Node,
) -> CResult<Node> {
    graph_two_nodes_method_helper(graph_ptr, a_ptr, b_ptr, |g, a, b| g.add(a, b))
}

#[no_mangle]
pub extern "C" fn graph_subtract(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    b_ptr: *mut Node,
) -> CResult<Node> {
    graph_two_nodes_method_helper(graph_ptr, a_ptr, b_ptr, |g, a, b| g.subtract(a, b))
}

#[no_mangle]
pub extern "C" fn graph_multiply(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    b_ptr: *mut Node,
) -> CResult<Node> {
    graph_two_nodes_method_helper(graph_ptr, a_ptr, b_ptr, |g, a, b| g.multiply(a, b))
}
#[no_mangle]
pub extern "C" fn graph_dot(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    b_ptr: *mut Node,
) -> CResult<Node> {
    graph_two_nodes_method_helper(graph_ptr, a_ptr, b_ptr, |g, a, b| g.dot(a, b))
}

#[no_mangle]
pub extern "C" fn graph_matmul(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    b_ptr: *mut Node,
) -> CResult<Node> {
    graph_two_nodes_method_helper(graph_ptr, a_ptr, b_ptr, |g, a, b| g.matmul(a, b))
}

#[no_mangle]
pub extern "C" fn graph_truncate(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    scale: u64,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| g.truncate(a, scale))
}

#[no_mangle]
pub extern "C" fn graph_sum(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    axis: CVecVal<u64>,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| {
        let axis_vec = axis.to_vec()?;
        g.sum(a, axis_vec)
    })
}

#[no_mangle]
pub extern "C" fn graph_permute_axes(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    axis: CVecVal<u64>,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| {
        let axis_vec = axis.to_vec()?;
        g.permute_axes(a, axis_vec)
    })
}

#[no_mangle]
pub extern "C" fn graph_get(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    index: CVecVal<u64>,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| {
        let index_vec = index.to_vec()?;
        g.get(a, index_vec)
    })
}

#[no_mangle]
pub extern "C" fn graph_get_slice(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    cslice: CSlice,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| {
        let slice = cslice.to_slice()?;
        g.get_slice(a, slice)
    })
}

#[no_mangle]
pub extern "C" fn graph_reshape(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    new_type_ptr: *mut Type,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| {
        let new_t = unsafe_deref(new_type_ptr)?;
        g.reshape(a, new_t)
    })
}

#[no_mangle]
pub extern "C" fn graph_random(graph_ptr: *mut Graph, output_type_ptr: *mut Type) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let output_t = unsafe_deref(output_type_ptr)?;
        g.random(output_t)
    })
}

#[no_mangle]
pub extern "C" fn graph_stack(
    graph_ptr: *mut Graph,
    nodes: CVec<Node>,
    outer_shape: CVecVal<u64>,
) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let nodes_vec = nodes.to_vec()?;
        let outer_shape_vec = outer_shape.to_vec()?;
        g.stack(nodes_vec, outer_shape_vec)
    })
}
#[no_mangle]
pub extern "C" fn graph_constant(graph_ptr: *mut Graph, typed_value: CTypedValue) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let (t, v) = typed_value.to_type_value()?;
        g.constant(t, v)
    })
}

#[no_mangle]
pub extern "C" fn graph_a2b(graph_ptr: *mut Graph, a_ptr: *mut Node) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| g.a2b(a))
}

#[no_mangle]
pub extern "C" fn graph_b2a(
    graph_ptr: *mut Graph,
    a_ptr: *mut Node,
    scalar_type_ptr: *mut ScalarType,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| {
        let st = unsafe_deref(scalar_type_ptr)?;
        g.b2a(a, st)
    })
}
#[no_mangle]
pub extern "C" fn graph_create_tuple(graph_ptr: *mut Graph, elements: CVec<Node>) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let elem = elements.to_vec()?;
        g.create_tuple(elem)
    })
}

#[no_mangle]
pub extern "C" fn graph_create_vector(
    graph_ptr: *mut Graph,
    type_ptr: *mut Type,
    elements: CVec<Node>,
) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let t = unsafe_deref(type_ptr)?;
        let elem = elements.to_vec()?;
        g.create_vector(t, elem)
    })
}

#[no_mangle]
pub extern "C" fn graph_create_named_tuple(
    graph_ptr: *mut Graph,
    elements_nodes: CVec<Node>,
    elements_names: CVecVal<CStr>,
) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let elem_nodes = elements_nodes.to_vec()?;
        let elem_names = elements_names.to_vec()?;
        let elem_names_string: Vec<String> = elem_names
            .iter()
            .map(|x| -> Result<String> { x.to_string() })
            .collect::<Result<Vec<String>>>()?;
        let elem: Vec<(String, Node)> = elem_names_string
            .iter()
            .zip(elem_nodes.iter())
            .map(|(x, y)| ((*x).clone(), (*y).clone()))
            .collect();
        g.create_named_tuple(elem)
    })
}
#[no_mangle]
pub extern "C" fn graph_tuple_get(
    graph_ptr: *mut Graph,
    tuple_node_ptr: *mut Node,
    index: u64,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, tuple_node_ptr, |g, t| g.tuple_get(t, index))
}

#[no_mangle]
pub extern "C" fn graph_named_tuple_get(
    graph_ptr: *mut Graph,
    tuple_node_ptr: *mut Node,
    key: CStr,
) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, tuple_node_ptr, |g, t| {
        let key_string = key.to_string()?;
        g.named_tuple_get(t, key_string)
    })
}

#[no_mangle]
pub extern "C" fn graph_vector_get(
    graph_ptr: *mut Graph,
    vec_node_ptr: *mut Node,
    index_node_ptr: *mut Node,
) -> CResult<Node> {
    graph_two_nodes_method_helper(graph_ptr, vec_node_ptr, index_node_ptr, |g, v, i| {
        g.vector_get(v, i)
    })
}

#[no_mangle]
pub extern "C" fn graph_zip(graph_ptr: *mut Graph, elements: CVec<Node>) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| g.zip(elements.to_vec()?))
}

#[no_mangle]
pub extern "C" fn graph_repeat(graph_ptr: *mut Graph, a_ptr: *mut Node, n: u64) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| g.repeat(a, n))
}

#[no_mangle]
pub extern "C" fn graph_call(
    graph_ptr: *mut Graph,
    callee_ptr: *mut Graph,
    arguments: CVec<Node>,
) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let callee = unsafe_deref(callee_ptr)?;
        g.call(callee, arguments.to_vec()?)
    })
}

#[no_mangle]
pub extern "C" fn graph_iterate(
    graph_ptr: *mut Graph,
    callee_ptr: *mut Graph,
    state_ptr: *mut Node,
    input_ptr: *mut Node,
) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        let callee = unsafe_deref(callee_ptr)?;
        let state = unsafe_deref(state_ptr)?;
        let input = unsafe_deref(input_ptr)?;
        g.iterate(callee, state, input)
    })
}

#[no_mangle]
pub extern "C" fn graph_vector_to_array(graph_ptr: *mut Graph, a_ptr: *mut Node) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| g.vector_to_array(a))
}

#[no_mangle]
pub extern "C" fn graph_array_to_vector(graph_ptr: *mut Graph, a_ptr: *mut Node) -> CResult<Node> {
    graph_one_node_method_helper(graph_ptr, a_ptr, |g, a| g.array_to_vector(a))
}

#[no_mangle]
pub extern "C" fn graph_custom_op(
    graph_ptr: *mut Graph,
    c_custom_op: CCustomOperation,
    args: CVec<Node>,
) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| {
        g.custom_op(c_custom_op.to_custom_op()?, args.to_vec()?)
    })
}

#[no_mangle]
pub extern "C" fn graph_finalize(graph_ptr: *mut Graph) -> CResult<Graph> {
    graph_method_helper(graph_ptr, |g| g.finalize())
}

#[no_mangle]
pub extern "C" fn graph_get_nodes(graph_ptr: *mut Graph) -> CResult<CVec<Node>> {
    graph_method_helper(graph_ptr, |g| Ok(CVec::from_vec(g.get_nodes())))
}

#[no_mangle]
pub extern "C" fn graph_set_output_node(
    graph_ptr: *mut Graph,
    n_ptr: *mut Node,
) -> CResultVal<bool> {
    graph_one_node_method_helper(graph_ptr, n_ptr, |g, n| {
        let res = g.set_output_node(n);
        match res {
            Ok(_) => Ok(true),
            Err(e) => Err(e),
        }
    })
}

#[no_mangle]
pub extern "C" fn graph_get_output_node(graph_ptr: *mut Graph) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| g.get_output_node())
}

#[no_mangle]
pub extern "C" fn graph_get_id(graph_ptr: *mut Graph) -> CResultVal<u64> {
    graph_method_helper(graph_ptr, |g| Ok(g.get_id()))
}

#[no_mangle]
pub extern "C" fn graph_get_num_nodes(graph_ptr: *mut Graph) -> CResultVal<u64> {
    graph_method_helper(graph_ptr, |g| Ok(g.get_num_nodes()))
}

#[no_mangle]
pub extern "C" fn graph_get_node_by_id(graph_ptr: *mut Graph, id: u64) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| g.get_node_by_id(id))
}

#[no_mangle]
pub extern "C" fn graph_get_context(graph_ptr: *mut Graph) -> CResult<Context> {
    graph_method_helper(graph_ptr, |g| Ok(g.get_context()))
}

#[no_mangle]
pub extern "C" fn graph_set_as_main(graph_ptr: *mut Graph) -> CResult<Graph> {
    graph_method_helper(graph_ptr, |g| g.set_as_main())
}

#[no_mangle]
pub extern "C" fn graph_set_name(graph_ptr: *mut Graph, name: CStr) -> CResult<Graph> {
    graph_method_helper(graph_ptr, |g| g.set_name(name.to_str_slice()?))
}

#[no_mangle]
pub extern "C" fn graph_get_name(graph_ptr: *mut Graph) -> CResultVal<CStr> {
    graph_method_helper(graph_ptr, |g| CStr::from_string(g.get_name()?))
}

#[no_mangle]
pub extern "C" fn graph_retrieve_node(graph_ptr: *mut Graph, name: CStr) -> CResult<Node> {
    graph_method_helper(graph_ptr, |g| g.retrieve_node(name.to_str_slice()?))
}

#[no_mangle]
pub extern "C" fn create_context() -> CResult<Context> {
    let context_res = ciphercore_base::graphs::create_context();
    CResult::new(context_res)
}

fn context_method_helper<T, F, R: CResultTrait<T>>(context_ptr: *mut Context, op: F) -> R
where
    F: FnOnce(Context) -> Result<T>,
{
    let helper = || -> Result<T> {
        let context = unsafe_deref(context_ptr)?;
        op(context)
    };
    R::new(helper())
}

#[no_mangle]
pub extern "C" fn context_create_graph(context_ptr: *mut Context) -> CResult<Graph> {
    context_method_helper(context_ptr, |c| c.create_graph())
}

#[no_mangle]
pub extern "C" fn context_finalize(context_ptr: *mut Context) -> CResult<Context> {
    context_method_helper(context_ptr, |c| c.finalize())
}

#[no_mangle]
pub extern "C" fn context_set_main_graph(
    context_ptr: *mut Context,
    graph_ptr: *mut Graph,
) -> CResult<Context> {
    context_method_helper(context_ptr, |c| {
        let graph = unsafe_deref(graph_ptr)?;
        c.set_main_graph(graph)
    })
}

#[no_mangle]
pub extern "C" fn context_get_graphs(context_ptr: *mut Context) -> CResult<CVec<Graph>> {
    context_method_helper(context_ptr, |c| Ok(CVec::from_vec(c.get_graphs())))
}

#[no_mangle]
pub extern "C" fn context_check_finalized(context_ptr: *mut Context) -> CResultVal<bool> {
    context_method_helper(context_ptr, |c| {
        let res = c.check_finalized();
        match res {
            Ok(_) => Ok(true),
            Err(e) => Err(e),
        }
    })
}

#[no_mangle]
pub extern "C" fn context_get_main_graph(context_ptr: *mut Context) -> CResult<Graph> {
    context_method_helper(context_ptr, |c| c.get_main_graph())
}

#[no_mangle]
pub extern "C" fn context_get_num_graphs(context_ptr: *mut Context) -> CResultVal<u64> {
    context_method_helper(context_ptr, |c| Ok(c.get_num_graphs()))
}

#[no_mangle]
pub extern "C" fn context_get_graph_by_id(context_ptr: *mut Context, id: u64) -> CResult<Graph> {
    context_method_helper(context_ptr, |c| c.get_graph_by_id(id))
}

#[no_mangle]
pub extern "C" fn context_get_node_by_global_id(
    context_ptr: *mut Context,
    global_id: CVecVal<u64>,
) -> CResult<Node> {
    context_method_helper(context_ptr, |c| {
        let g_id_vec = global_id.to_vec()?;
        if g_id_vec.len() != 2 {
            return Err(runtime_error!("Global Id vector should have two elements!"));
        }
        c.get_node_by_global_id((g_id_vec[0], g_id_vec[1]))
    })
}

#[no_mangle]
pub extern "C" fn context_to_string(context_ptr: *mut Context) -> CResultVal<CStr> {
    context_method_helper(context_ptr, |c| {
        CStr::from_string(serde_json::to_string(&c)?)
    })
}

#[no_mangle]
pub extern "C" fn contexts_deep_equal(
    context1_ptr: *mut Context,
    context2_ptr: *mut Context,
) -> CResultVal<bool> {
    context_method_helper(context1_ptr, |c| {
        let c2 = unsafe_deref(context2_ptr)?;
        Ok(ciphercore_base::graphs::contexts_deep_equal(c, c2))
    })
}

#[no_mangle]
pub extern "C" fn context_set_graph_name(
    context_ptr: *mut Context,
    graph_ptr: *mut Graph,
    name: CStr,
) -> CResult<Context> {
    context_method_helper(context_ptr, |c| {
        let graph = unsafe_deref(graph_ptr)?;
        c.set_graph_name(graph, name.to_str_slice()?)
    })
}

#[no_mangle]
pub extern "C" fn context_get_graph_name(
    context_ptr: *mut Context,
    graph_ptr: *mut Graph,
) -> CResultVal<CStr> {
    context_method_helper(context_ptr, |c| {
        let graph = unsafe_deref(graph_ptr)?;
        CStr::from_string(c.get_graph_name(graph)?)
    })
}
#[no_mangle]
pub extern "C" fn context_retrieve_graph(context_ptr: *mut Context, name: CStr) -> CResult<Graph> {
    context_method_helper(context_ptr, |c| c.retrieve_graph(name.to_str_slice()?))
}

#[no_mangle]
pub extern "C" fn context_set_node_name(
    context_ptr: *mut Context,
    node_ptr: *mut Node,
    name: CStr,
) -> CResult<Context> {
    context_method_helper(context_ptr, |c| {
        let node = unsafe_deref(node_ptr)?;
        c.set_node_name(node, name.to_str_slice()?)
    })
}

#[no_mangle]
pub extern "C" fn context_get_node_name(
    context_ptr: *mut Context,
    node_ptr: *mut Node,
) -> CResultVal<CStr> {
    context_method_helper(context_ptr, |c| {
        let node = unsafe_deref(node_ptr)?;
        CStr::from_string(c.get_node_name(node)?)
    })
}

#[no_mangle]
pub extern "C" fn context_retrieve_node(
    context_ptr: *mut Context,
    graph_ptr: *mut Graph,
    name: CStr,
) -> CResult<Node> {
    context_method_helper(context_ptr, |c| {
        let graph = unsafe_deref(graph_ptr)?;
        c.retrieve_node(graph, name.to_str_slice()?)
    })
}

#[no_mangle]
pub extern "C" fn context_destroy(context_ptr: *mut Context) -> () {
    destroy_helper(context_ptr);
}
#[no_mangle]
pub extern "C" fn graph_destroy(graph_ptr: *mut Graph) -> () {
    destroy_helper(graph_ptr);
}

#[no_mangle]
pub extern "C" fn node_destroy(node_ptr: *mut Node) -> () {
    destroy_helper(node_ptr);
}

fn node_method_helper<T, F, R: CResultTrait<T>>(node_ptr: *mut Node, op: F) -> R
where
    F: FnOnce(Node) -> Result<T>,
{
    let helper = || -> Result<T> {
        let node = unsafe_deref(node_ptr)?;
        op(node)
    };
    R::new(helper())
}
fn node_one_node_method_helper<T, F, R: CResultTrait<T>>(
    node_ptr: *mut Node,
    b_ptr: *mut Node,
    op: F,
) -> R
where
    F: FnOnce(Node, Node) -> Result<T>,
{
    let helper = || -> Result<T> {
        let node = unsafe_deref(node_ptr)?;
        let b = unsafe_deref(b_ptr)?;
        op(node, b)
    };
    R::new(helper())
}
#[no_mangle]
pub extern "C" fn node_get_graph(node_ptr: *mut Node) -> CResult<Graph> {
    node_method_helper(node_ptr, |n| Ok(n.get_graph()))
}

#[no_mangle]
pub extern "C" fn node_get_dependencies(node_ptr: *mut Node) -> CResult<CVec<Node>> {
    node_method_helper(node_ptr, |n| Ok(CVec::from_vec(n.get_node_dependencies())))
}

#[no_mangle]
pub extern "C" fn node_graph_dependencies(node_ptr: *mut Node) -> CResult<CVec<Graph>> {
    node_method_helper(node_ptr, |n| Ok(CVec::from_vec(n.get_graph_dependencies())))
}
#[no_mangle]
pub extern "C" fn node_get_operation(node_ptr: *mut Node) -> CResult<COperation> {
    node_method_helper(node_ptr, |n| COperation::from_operation(n.get_operation()))
}

#[no_mangle]
pub extern "C" fn node_get_id(node_ptr: *mut Node) -> CResultVal<u64> {
    node_method_helper(node_ptr, |n| Ok(n.get_id()))
}

#[no_mangle]
pub extern "C" fn node_get_global_id(node_ptr: *mut Node) -> CResult<CVecVal<u64>> {
    node_method_helper(node_ptr, |n| {
        let ids = n.get_global_id();
        let ids_vec = vec![ids.0, ids.1];
        Ok(CVecVal::from_vec(ids_vec))
    })
}

#[no_mangle]
pub extern "C" fn node_get_type(node_ptr: *mut Node) -> CResult<Type> {
    node_method_helper(node_ptr, |n| n.get_type())
}

#[no_mangle]
pub extern "C" fn node_add(node_ptr: *mut Node, b_ptr: *mut Node) -> CResult<Node> {
    node_one_node_method_helper(node_ptr, b_ptr, |a, b| a.add(b))
}

#[no_mangle]
pub extern "C" fn node_subtract(node_ptr: *mut Node, b_ptr: *mut Node) -> CResult<Node> {
    node_one_node_method_helper(node_ptr, b_ptr, |a, b| a.subtract(b))
}

#[no_mangle]
pub extern "C" fn node_multiply(node_ptr: *mut Node, b_ptr: *mut Node) -> CResult<Node> {
    node_one_node_method_helper(node_ptr, b_ptr, |a, b| a.multiply(b))
}
#[no_mangle]
pub extern "C" fn node_dot(node_ptr: *mut Node, b_ptr: *mut Node) -> CResult<Node> {
    node_one_node_method_helper(node_ptr, b_ptr, |a, b| a.dot(b))
}

#[no_mangle]
pub extern "C" fn node_matmul(node_ptr: *mut Node, b_ptr: *mut Node) -> CResult<Node> {
    node_one_node_method_helper(node_ptr, b_ptr, |a, b| a.matmul(b))
}

#[no_mangle]
pub extern "C" fn node_truncate(node_ptr: *mut Node, scale: u64) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.truncate(scale))
}

#[no_mangle]
pub extern "C" fn node_sum(node_ptr: *mut Node, axis: CVecVal<u64>) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.sum(axis.to_vec()?))
}
#[no_mangle]
pub extern "C" fn node_permute_axes(node_ptr: *mut Node, axis: CVecVal<u64>) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.permute_axes(axis.to_vec()?))
}

#[no_mangle]
pub extern "C" fn node_get(node_ptr: *mut Node, index: CVecVal<u64>) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.get(index.to_vec()?))
}

#[no_mangle]
pub extern "C" fn node_get_slice(node_ptr: *mut Node, cslice: CSlice) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.get_slice(cslice.to_slice()?))
}

#[no_mangle]
pub extern "C" fn node_reshape(node_ptr: *mut Node, type_ptr: *mut Type) -> CResult<Node> {
    node_method_helper(node_ptr, |a| {
        let t = unsafe_deref(type_ptr)?;
        a.reshape(t)
    })
}

#[no_mangle]
pub extern "C" fn node_nop(node_ptr: *mut Node) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.nop())
}

#[no_mangle]
pub extern "C" fn node_prf(
    node_ptr: *mut Node,
    iv: u64,
    output_type_ptr: *mut Type,
) -> CResult<Node> {
    node_method_helper(node_ptr, |a| {
        let output_type = unsafe_deref(output_type_ptr)?;
        a.prf(iv, output_type)
    })
}

#[no_mangle]
pub extern "C" fn node_a2b(node_ptr: *mut Node) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.a2b())
}

#[no_mangle]
pub extern "C" fn node_b2a(node_ptr: *mut Node, scalar_type_ptr: *mut ScalarType) -> CResult<Node> {
    node_method_helper(node_ptr, |a| {
        let st = unsafe_deref(scalar_type_ptr)?;
        a.b2a(st)
    })
}

#[no_mangle]
pub extern "C" fn node_tuple_get(node_ptr: *mut Node, index: u64) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.tuple_get(index))
}

#[no_mangle]
pub extern "C" fn node_named_tuple_get(node_ptr: *mut Node, key: CStr) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.named_tuple_get(key.to_string()?))
}

#[no_mangle]
pub extern "C" fn node_vector_get(node_ptr: *mut Node, index_node_ptr: *mut Node) -> CResult<Node> {
    node_one_node_method_helper(node_ptr, index_node_ptr, |a, index| a.vector_get(index))
}

#[no_mangle]
pub extern "C" fn node_array_to_vector(node_ptr: *mut Node) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.array_to_vector())
}

#[no_mangle]
pub extern "C" fn node_vector_to_array(node_ptr: *mut Node) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.vector_to_array())
}

#[no_mangle]
pub extern "C" fn node_repeat(node_ptr: *mut Node, n: u64) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.repeat(n))
}

#[no_mangle]
pub extern "C" fn node_set_as_output(node_ptr: *mut Node) -> CResult<Node> {
    node_method_helper(node_ptr, |a| a.set_as_output())
}
