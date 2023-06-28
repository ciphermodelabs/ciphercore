use crate::bytes::vec_u64_from_bytes;
use crate::custom_ops::ContextMappings;
use crate::data_types::UINT64;
use crate::errors::Result;
use crate::graphs::{copy_node_name, Graph, Node, Operation, SliceElement};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq)]
enum ProxyObject {
    Number(u64),
    UnknownNode,
    ArrayToVector(Node),
    Tuple(Vec<Arc<ProxyObjectWithNode>>),
    NamedTuple(HashMap<String, Arc<ProxyObjectWithNode>>),
    Zip(Vec<Arc<ProxyObjectWithNode>>),
    Vector(Vec<Arc<ProxyObjectWithNode>>),
    A2B(Node),
    B2A(Node),
}

#[derive(Clone, PartialEq, Eq)]
struct ProxyObjectWithNode {
    meta: ProxyObject,
    node: Node,
}

/// In this optimizer, we maintain proxy objects for nodes created with meta operations
/// (create_tuple/create_vector/array_to_vector). It then applies natural simplifications
/// (e.g. tuple_get(create_tuple(a, b), 0) = a). Of course it is not always possible
/// to eliminate these meta operations (some common examples include create_vector + vector_to_array,
/// and create_tuple + reshape). But the subgraphs consisting only of these meta-operations
/// and their getters can be optimized.
/// This process leaves a bunch of dangling nodes, which then can be removed later.
/// This function preserves annotations and node names of remaining nodes.
/// If a node or a group of nodes is replaced by another node, the related names are not preserved.
pub(super) fn optimize_graph_meta_operations(
    graph: Graph,
    out_graph: Graph,
) -> Result<ContextMappings> {
    let mut mapping = ContextMappings::default();
    let mut meta_objects = HashMap::<Node, ProxyObjectWithNode>::new();
    for node in graph.get_nodes() {
        if !node.get_graph_dependencies().is_empty() {
            return Err(runtime_error!(
                "Meta-operation optimization works only on fully inlined graphs."
            ));
        }
        let mut deps = vec![];
        let mut meta_deps = vec![];
        let mut all_meta_deps = true;
        for dep in node.get_node_dependencies() {
            deps.push(mapping.get_node(&dep));

            let meta_dep = meta_objects.get(&dep);
            match meta_dep {
                Some(resolved_meta_dep) => {
                    meta_deps.push(Some(resolved_meta_dep.clone()));
                }
                None => {
                    meta_deps.push(None);
                    all_meta_deps = false;
                }
            }
        }
        let simple_node = out_graph.add_node_with_type(
            deps.clone(),
            vec![],
            node.get_operation(),
            node.get_type()?,
        )?;
        copy_node_name(node.clone(), simple_node.clone())?;
        let meta_node = match node.get_operation() {
            Operation::Constant(t, v) => {
                // Note: for constants we're only caring about UINT64, since these can be
                // used to index vectors.
                // TODO: update once VectorGet supports other types.
                if t.is_scalar() && t.get_scalar_type() == UINT64 {
                    v.access_bytes(|bytes| {
                        Ok(Some(ProxyObjectWithNode {
                            meta: ProxyObject::Number(vec_u64_from_bytes(bytes, UINT64)?[0]),
                            node: simple_node.clone(),
                        }))
                    })?
                } else {
                    None
                }
            }
            Operation::ArrayToVector => Some(ProxyObjectWithNode {
                meta: ProxyObject::ArrayToVector(deps[0].clone()),
                node: simple_node.clone(),
            }),
            Operation::A2B => {
                let mut node = simple_node.clone();
                if let Some(meta_dep) = &meta_deps[0] {
                    if let ProxyObject::B2A(binary_node) = &meta_dep.meta {
                        node = binary_node.clone();
                    }
                }
                Some(ProxyObjectWithNode {
                    meta: ProxyObject::A2B(deps[0].clone()),
                    node,
                })
            }
            Operation::B2A(st) => {
                let mut node = simple_node.clone();
                if let Some(meta_dep) = &meta_deps[0] {
                    if let ProxyObject::A2B(arithmetic_node) = &meta_dep.meta {
                        if st == arithmetic_node.get_type()?.get_scalar_type() {
                            node = arithmetic_node.clone();
                        }
                    }
                }
                Some(ProxyObjectWithNode {
                    meta: ProxyObject::B2A(deps[0].clone()),
                    node,
                })
            }
            Operation::CreateNamedTuple(names) => {
                let mut computed_elements = HashMap::new();
                for i in 0..deps.len() {
                    let element = if let Some(meta) = meta_deps[i].clone() {
                        meta.clone()
                    } else {
                        ProxyObjectWithNode {
                            meta: ProxyObject::UnknownNode,
                            node: deps[i].clone(),
                        }
                    };
                    computed_elements.insert(names[i].clone(), Arc::new(element));
                }
                Some(ProxyObjectWithNode {
                    meta: ProxyObject::NamedTuple(computed_elements),
                    node: simple_node.clone(),
                })
            }
            Operation::CreateTuple => {
                let mut computed_elements = vec![];
                for i in 0..deps.len() {
                    let element = if let Some(meta) = meta_deps[i].clone() {
                        meta.clone()
                    } else {
                        ProxyObjectWithNode {
                            meta: ProxyObject::UnknownNode,
                            node: deps[i].clone(),
                        }
                    };
                    computed_elements.push(Arc::new(element));
                }
                Some(ProxyObjectWithNode {
                    meta: ProxyObject::Tuple(computed_elements),
                    node: simple_node.clone(),
                })
            }
            Operation::CreateVector(_) => {
                let mut computed_elements = vec![];
                for i in 0..deps.len() {
                    let element = if let Some(meta) = meta_deps[i].clone() {
                        meta.clone()
                    } else {
                        ProxyObjectWithNode {
                            meta: ProxyObject::UnknownNode,
                            node: deps[i].clone(),
                        }
                    };
                    computed_elements.push(Arc::new(element));
                }
                Some(ProxyObjectWithNode {
                    meta: ProxyObject::Vector(computed_elements),
                    node: simple_node.clone(),
                })
            }
            Operation::Zip => {
                let mut computed_elements = vec![];
                for i in 0..deps.len() {
                    let element = if let Some(meta) = meta_deps[i].clone() {
                        meta.clone()
                    } else {
                        ProxyObjectWithNode {
                            meta: ProxyObject::UnknownNode,
                            node: deps[i].clone(),
                        }
                    };
                    computed_elements.push(Arc::new(element));
                }
                Some(ProxyObjectWithNode {
                    meta: ProxyObject::Zip(computed_elements),
                    node: simple_node.clone(),
                })
            }
            _ => {
                if all_meta_deps {
                    let unwrapped_deps: Vec<ProxyObjectWithNode> = meta_deps
                        .iter()
                        .map(|x| x.as_ref().unwrap().clone())
                        .collect();
                    maybe_apply_meta_op(out_graph.clone(), node.clone(), unwrapped_deps)?
                } else {
                    None
                }
            }
        };
        let new_node = if let Some(actual_meta_node) = meta_node {
            meta_objects.insert(node.clone(), actual_meta_node.clone());
            actual_meta_node.node.clone()
        } else {
            simple_node
        };
        for annotation in node.get_annotations()? {
            new_node.add_annotation(annotation)?;
        }
        if node == graph.get_output_node()? {
            new_node.set_as_output()?;
        }
        mapping.insert_node(node, new_node);
    }
    Ok(mapping)
}

/// Tries to apply an operation while simplifying the graph.
/// Returns None if simplifications are not possible.
fn maybe_apply_meta_op(
    graph: Graph,
    node: Node,
    deps: Vec<ProxyObjectWithNode>,
) -> Result<Option<ProxyObjectWithNode>> {
    match node.get_operation() {
        Operation::NamedTupleGet(name) => {
            if deps.len() != 1 {
                return Err(runtime_error!("NamedTupleGet should have 1 argument"));
            }
            if let ProxyObject::NamedTuple(element_ptrs) = deps[0].clone().meta {
                Ok(Some((*element_ptrs[&name]).clone()))
            } else {
                Ok(None)
            }
        }
        Operation::TupleGet(index) => {
            if deps.len() != 1 {
                return Err(runtime_error!("TupleGet should have 1 argument"));
            }
            if let ProxyObject::Tuple(element_ptrs) = deps[0].clone().meta {
                Ok(Some((*element_ptrs[index as usize]).clone()))
            } else {
                Ok(None)
            }
        }
        Operation::VectorGet => {
            if deps.len() != 2 {
                return Err(runtime_error!("VectorGet should have 1 argument"));
            }
            if let ProxyObject::Number(index) = deps[1].meta {
                maybe_vector_get(graph, deps[0].clone(), index, deps[1].node.clone())
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Tries to simplify VectorGet. The catch is, this function is recursive.
/// This is needed for the case when one calls VectorGet on the output of Zip
/// (common case in comparisons).
fn maybe_vector_get(
    graph: Graph,
    meta_obj: ProxyObjectWithNode,
    index: u64,
    index_node: Node,
) -> Result<Option<ProxyObjectWithNode>> {
    match meta_obj.meta {
        ProxyObject::Vector(element_ptrs) => Ok(Some((*element_ptrs[index as usize]).clone())),
        ProxyObject::ArrayToVector(arr_node) => {
            // VectorGet is replaced by Get or GetSlice.
            // The name of the VectorGet node is passed to a new node.
            let new_node = if arr_node.get_type()?.get_dimensions().len() == 1 {
                arr_node.get(vec![index])?
            } else {
                arr_node.get_slice(vec![
                    SliceElement::SingleIndex(index as i64),
                    SliceElement::Ellipsis,
                ])?
            };

            Ok(Some(ProxyObjectWithNode {
                meta: ProxyObject::UnknownNode,
                node: new_node,
            }))
        }
        ProxyObject::UnknownNode => {
            let new_node = meta_obj.node.vector_get(index_node)?;
            Ok(Some(ProxyObjectWithNode {
                meta: ProxyObject::UnknownNode,
                node: new_node,
            }))
        }
        ProxyObject::Zip(vecs) => {
            let mut sliced_elements = vec![];
            let mut success = true;
            for vec in vecs {
                let maybe_slice =
                    maybe_vector_get(graph.clone(), (*vec).clone(), index, index_node.clone())?;
                if let Some(slice) = maybe_slice {
                    sliced_elements.push(slice.clone());
                } else {
                    success = false;
                    break;
                }
            }
            if success {
                let element_nodes: Vec<Node> =
                    sliced_elements.iter().map(|e| e.node.clone()).collect();
                let element_ptrs: Vec<Arc<ProxyObjectWithNode>> = sliced_elements
                    .iter()
                    .map(|e| Arc::new(e.clone()))
                    .collect();
                let new_node = graph.create_tuple(element_nodes)?;
                Ok(Some(ProxyObjectWithNode {
                    meta: ProxyObject::Tuple(element_ptrs),
                    node: new_node,
                }))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{array_type, scalar_type, BIT, INT64, UINT64};
    use crate::data_values::Value;
    use crate::graphs::contexts_deep_equal;
    use crate::graphs::util::simple_context;
    use crate::graphs::{create_context, Context};
    use crate::optimizer::dangling_nodes_optimizer::optimize_graph_dangling_nodes;

    fn optimize_helper(c: Context) -> Result<Context> {
        let new_c1 = create_context()?;
        let new_g1 = new_c1.create_graph()?;
        optimize_graph_meta_operations(c.get_main_graph()?, new_g1.clone())?;
        new_g1.finalize()?;
        new_g1.set_as_main()?;
        new_c1.finalize()?;
        let new_c2 = create_context()?;
        let new_g2 = new_c2.create_graph()?;
        optimize_graph_dangling_nodes(new_c1.get_main_graph()?, new_g2.clone())?;
        new_g2.finalize()?;
        new_g2.set_as_main()?;
        new_c2.finalize()?;
        Ok(new_c2)
    }

    #[test]
    fn test_no_meta() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n = i1.add(i2)?;
                n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)
            })?;

            let new_c = optimize_helper(c.clone())?;
            assert!(contexts_deep_equal(&new_c, &c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_simple_tuple_get() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let t = g.create_tuple(vec![i1, i2])?;
                t.set_name("CreateTuple")?;
                let o = t.tuple_get(0)?;
                o.set_name("TupleGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let _i2 = g.input(scalar_type(UINT64))?;
                Ok(i1)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_nested_tuple_get() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let t1 = g.create_tuple(vec![i1.clone(), i2])?;
                let t2 = g.create_tuple(vec![t1, i1])?;
                let o1 = t2.tuple_get(0)?;
                o1.set_name("First TupleGet")?;
                let o = o1.tuple_get(1)?;
                o.set_name("Second TupleGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let _i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                Ok(i2)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_simple_named_tuple_get() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n1 = "n1".to_string();
                let n2 = "n2".to_string();
                let t = g.create_named_tuple(vec![(n1.clone(), i1), (n2, i2)])?;
                t.set_name("CreateNamedTuple")?;
                let o = t.named_tuple_get(n1)?;
                o.set_name("NamedTupleGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let _i2 = g.input(scalar_type(UINT64))?;
                Ok(i1)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_nested_named_tuple_get() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n1 = "n1".to_string();
                let n2 = "n2".to_string();
                let n3 = "n3".to_string();
                let t1 = g.create_named_tuple(vec![(n1.clone(), i1.clone()), (n2.clone(), i2)])?;
                let t2 = g.create_named_tuple(vec![(n3.clone(), t1), (n1, i1)])?;
                let o1 = t2.named_tuple_get(n3)?;
                o1.set_name("First NamedTupleGet")?;
                let o = o1.named_tuple_get(n2)?;
                o.set_name("Second NamedTupleGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let _i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                Ok(i2)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_simple_vector_get() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                i1.set_name("Input")?;
                let i2 = g.input(scalar_type(UINT64))?;
                let v = g.create_vector(i1.get_type()?, vec![i1, i2])?;
                v.set_name("CreateVector")?;
                let o =
                    v.vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(0, UINT64)?)?)?;
                o.set_name("VectorGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let _i2 = g.input(scalar_type(UINT64))?;
                let o = i1;
                o.set_name("Input")?;
                Ok(o)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_complex_tree() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                i1.set_name("Input")?;
                let i2 = g.input(scalar_type(UINT64))?;
                let v1 = g.create_vector(i1.get_type()?, vec![i1.clone(), i2.clone()])?;
                v1.set_name("CreateVector1")?;
                let v2 = g.create_vector(v1.get_type()?, vec![v1.clone(), v1.clone()])?;
                v2.set_name("CreateVector2")?;
                let t1 = g.create_tuple(vec![i1, v1.clone(), i2, v1])?;
                let t2 = g.create_tuple(vec![t1, v2.clone()])?;
                let o1 = t2
                    .tuple_get(0)?
                    .tuple_get(1)?
                    .vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)?;
                let o2 = v2
                    .vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(0, UINT64)?)?)?;
                o2.set_name("VectorGet")?;
                let o = g.create_tuple(vec![o1, o2])?;
                o.set_name("CreateTuple")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                i1.set_name("Input")?;
                let i2 = g.input(scalar_type(UINT64))?;
                let o1 = i2.clone();
                let o2 = g.create_vector(i1.get_type()?, vec![i1, i2])?;
                o2.set_name("CreateVector1")?;
                let o = g.create_tuple(vec![o1, o2])?;
                o.set_name("CreateTuple")?;
                Ok(o)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_array_to_vector_get() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(array_type(vec![10], UINT64))?;
                i.set_name("Input")?;
                let v = i.array_to_vector()?;
                v.set_name("ArrayToVector")?;
                let o =
                    v.vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(0, UINT64)?)?)?;
                o.set_name("VectorGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let i = g.input(array_type(vec![10], UINT64))?;
                i.set_name("Input")?;
                i.get(vec![0])
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_array_to_vector_get_multiple_dims() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(array_type(vec![10, 2], UINT64))?;
                i.set_name("Input")?;
                let v = i.array_to_vector()?;
                v.set_name("ArrayToVector")?;
                let o =
                    v.vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(0, UINT64)?)?)?;
                o.set_name("VectorGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let i = g.input(array_type(vec![10, 2], UINT64))?;
                i.set_name("Input")?;
                i.get_slice(vec![SliceElement::SingleIndex(0), SliceElement::Ellipsis])
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_zip_arrays_get() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(array_type(vec![10], UINT64))?;
                i1.set_name("Input1")?;
                let i2 = g.input(array_type(vec![10], UINT64))?;
                i2.set_name("Input2")?;
                let v1 = i1.array_to_vector()?;
                v1.set_name("ArrayToVector1")?;
                let v2 = i2.array_to_vector()?;
                v2.set_name("ArrayToVector2")?;
                let v = g.zip(vec![v1, v2])?;
                v.set_name("Zip")?;
                let o1 =
                    v.vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?)?;
                o1.set_name("VectorGet")?;
                let o = o1.tuple_get(1)?;
                o.set_name("TupleGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let _i1 = g.input(array_type(vec![10], UINT64))?;
                _i1.set_name("Input1")?;
                let i2 = g.input(array_type(vec![10], UINT64))?;
                i2.set_name("Input2")?;
                i2.get(vec![2])
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_zip_unknown_arrays_get() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(array_type(vec![10], UINT64))?;
                i1.set_name("Input1")?;
                let i2 = g.input(array_type(vec![10], UINT64))?;
                i2.set_name("Input2")?;
                let v10 = i1.array_to_vector()?;
                v10.set_name("ArrayToVector1")?;
                let v20 = i2.array_to_vector()?;
                v20.set_name("ArrayToVector2")?;
                let v1 = v10.reshape(v10.get_type()?)?;
                v1.set_name("Reshape1")?;
                let v2 = v20.reshape(v20.get_type()?)?;
                v2.set_name("Reshape2")?;
                let v = g.zip(vec![v1, v2])?;
                v.set_name("Zip")?;
                let o1 =
                    v.vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?)?;
                o1.set_name("VectorGet")?;
                let o = o1.tuple_get(1)?;
                o.set_name("TupleGet")?;
                Ok(o)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let _i1 = g.input(array_type(vec![10], UINT64))?;
                _i1.set_name("Input1")?;
                let i2 = g.input(array_type(vec![10], UINT64))?;
                i2.set_name("Input2")?;
                let v20 = i2.array_to_vector()?;
                v20.set_name("ArrayToVector2")?;
                let v2 = v20.reshape(v20.get_type()?)?;
                v2.set_name("Reshape2")?;
                v2.vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_b2a_a2b() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(array_type(vec![16, 64], BIT))?;
                i.set_name("Input")?;
                let a = i.b2a(UINT64)?;
                a.a2b()
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let i = g.input(array_type(vec![16, 64], BIT))?;
                i.set_name("Input")?;
                Ok(i)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_a2b_b2a() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(array_type(vec![16], UINT64))?;
                i.set_name("Input")?;
                let a = i.a2b()?;
                a.b2a(UINT64)
            })?;
            let new_c = optimize_helper(c)?;

            let expected_c = simple_context(|g| {
                let i = g.input(array_type(vec![16], UINT64))?;
                i.set_name("Input")?;
                Ok(i)
            })?;
            assert!(contexts_deep_equal(&new_c, &expected_c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_a2b_b2a_different_type() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(array_type(vec![16], UINT64))?;
                i.set_name("Input")?;
                let a = i.a2b()?;
                a.b2a(INT64)
            })?;
            let new_c = optimize_helper(c.clone())?;

            assert!(contexts_deep_equal(&new_c, &c));
            Ok(())
        }()
        .unwrap();
    }
}
