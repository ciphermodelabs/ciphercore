use crate::errors::Result;
use crate::graphs::{copy_node_name, Graph, Node, NodeAnnotation, Operation};
use std::cmp::Eq;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
struct NodeKey {
    deps: Vec<u64>,
    annotations: Vec<NodeAnnotation>,
    op: Operation,
}

impl NodeKey {
    pub fn new(node: Node, dep_ids: Vec<u64>) -> Result<Option<Self>> {
        let op = node.get_operation();
        if op.is_prf_operation() || op.is_randomizing()? || op.is_input() {
            // Don't try to de-duplicate PRF, randomizing and input operations.
            return Ok(None);
        }
        match op {
            Operation::Constant(_, _) => {
                // Don't try to de-duplicate constant.
                Ok(None)
            }
            Operation::Custom(_) => Err(runtime_error!(
                "Graph has to be fully inlined for the duplicates optimization"
            )),
            _ => Ok(Some(NodeKey {
                deps: dep_ids,
                annotations: node.get_annotations()?,
                op: node.get_operation(),
            })),
        }
    }
}

fn hash_some_operations<H: Hasher>(op: &Operation, state: &mut H) {
    match op {
        // Ignore constants.
        Operation::Constant(_, _) => {}
        _ => {
            let string_repr = serde_json::to_string(op).unwrap();
            string_repr.hash(state);
        }
    }
}

impl Hash for NodeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deps.hash(state);
        self.annotations.hash(state);
        // Hash is not implemented for Operation, and the blocker is - Constant has Value inside,
        // which is not hashable. So we either do a custom implementation of Hash for Operation,
        // or rely on some simple hack: use JSON representation for hashing.
        // Since all operations except Constant are serialized to something small, this is OK.
        hash_some_operations(&self.op, state);
    }
}

impl PartialEq for NodeKey {
    fn eq(&self, other: &Self) -> bool {
        self.deps == other.deps && self.annotations == other.annotations && self.op == other.op
    }
}

impl Eq for NodeKey {}

/// This optimization assumes that the graph is fully inlined.
/// It traverses the nodes in order, eliminating duplicates (same dependencies and annotations).
/// For node names, the name of the first occurrence of the node is taken.
pub(super) fn optimize_graph_duplicates(graph: Graph, out_graph: Graph) -> Result<()> {
    graph.check_finalized()?;
    let mut node_mapping = HashMap::<Node, Node>::new();
    let mut node_signatures = HashMap::<NodeKey, Node>::new();
    for node in graph.get_nodes() {
        if !node.get_graph_dependencies().is_empty() {
            return Err(runtime_error!(
                "Duplicate optimization works only on fully inlined graphs."
            ));
        }
        let mut deps = vec![];
        let mut dep_ids = vec![];
        for dep in node.get_node_dependencies() {
            let new_dep = node_mapping.get(&dep).unwrap();
            deps.push(new_dep.clone());
            dep_ids.push(new_dep.get_id());
        }
        let node_key = NodeKey::new(node.clone(), dep_ids)?;
        let maybe_new_node = if let Some(key) = &node_key {
            node_signatures.get(key)
        } else {
            None
        };

        let new_node = if let Some(new_node) = maybe_new_node {
            new_node.clone()
        } else {
            let new_node = out_graph.add_node(deps, vec![], node.get_operation())?;
            for annotation in node.get_annotations()? {
                new_node.add_annotation(annotation)?;
            }
            copy_node_name(node.clone(), new_node.clone())?;
            new_node
        };

        if node == graph.get_output_node()? {
            new_node.set_as_output()?;
        }
        node_mapping.insert(node, new_node.clone());
        if let Some(key) = &node_key {
            node_signatures.insert(key.clone(), new_node);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{scalar_type, UINT64};
    use crate::data_values::Value;
    use crate::graphs::contexts_deep_equal;
    use crate::graphs::create_context;
    use crate::graphs::util::simple_context;

    #[test]
    fn test_no_dups() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n = i1.add(i2)?;
                let r1 = g.random(scalar_type(UINT64))?;
                let r2 = g.random_permutation(5)?;
                let r3 = r2.cuckoo_to_permutation()?;
                let r4 = r2.decompose_switching_map(5)?;
                let o = n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)?;
                g.create_tuple(vec![o, r1, r3, r4])
            })?;

            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_duplicates(c.get_main_graph()?.clone(), new_g.clone())?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;
            assert!(contexts_deep_equal(new_c, c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_some_dups() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n1 = i1.add(i2.clone())?;
                n1.set_name("node1")?;
                let n2 = i1.add(i2)?;
                n2.set_name("node2")?;
                let o1 = n1.add(n2.clone())?;
                let o2 = n2.add(n1)?;
                o1.add(o2)
            })?;

            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_duplicates(c.get_main_graph()?.clone(), new_g.clone())?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;

            assert!(!contexts_deep_equal(new_c.clone(), c.clone()));
            assert_eq!(c.get_main_graph()?.get_nodes().len(), 7);
            assert_eq!(new_c.get_main_graph()?.get_nodes().len(), 5);
            // Check names
            let new_node2 = new_c.retrieve_node(new_g.clone(), "node2");
            assert!(new_node2.is_err());
            let new_node1 = new_c.retrieve_node(new_g, "node1");
            assert!(new_node1.is_ok());
            assert_eq!(new_node1?.get_operation(), Operation::Add);
            Ok(())
        }()
        .unwrap();
    }
}
