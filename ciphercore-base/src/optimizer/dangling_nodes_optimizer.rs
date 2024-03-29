use crate::custom_ops::ContextMappings;
use crate::errors::Result;
use crate::graphs::{copy_node_name, Graph, Node};
use std::collections::HashSet;

/// This optimization removes the nodes from which the output node is not reachable.
/// This can happen if the graph is constructed inefficiently, or (more common) as a result
/// of other graph optimizations.
/// This function preserves annotations and names of remaining nodes.
pub(super) fn optimize_graph_dangling_nodes(
    graph: Graph,
    out_graph: Graph,
) -> Result<ContextMappings> {
    graph.check_finalized()?;
    let mut useful_nodes = HashSet::<Node>::new();
    useful_nodes.insert(graph.get_output_node()?);
    for node in graph.get_nodes().iter().rev() {
        if useful_nodes.contains(node) {
            for dep in node.get_node_dependencies() {
                useful_nodes.insert(dep.clone());
            }
        }
    }
    let mut mapping = ContextMappings::default();
    for node in graph.get_nodes() {
        if !node.get_operation().is_input() && !useful_nodes.contains(&node) {
            continue;
        }
        let mut deps = vec![];
        for dep in node.get_node_dependencies() {
            deps.push(mapping.get_node(&dep));
        }
        if !node.get_graph_dependencies().is_empty() {
            return Err(runtime_error!(
                "Graph must be fully inlined to use the optimizer"
            ));
        }
        let new_node =
            out_graph.add_node_with_type(deps, vec![], node.get_operation(), node.get_type()?)?;
        for annotation in node.get_annotations()? {
            new_node.add_annotation(annotation)?;
        }
        copy_node_name(node.clone(), new_node.clone())?;
        if node == graph.get_output_node()? {
            new_node.set_as_output()?;
        }
        mapping.insert_node(node, new_node);
    }
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{scalar_type, UINT64};
    use crate::data_values::Value;
    use crate::graphs::create_context;
    use crate::graphs::util::simple_context;
    use crate::graphs::{contexts_deep_equal, Operation};

    #[test]
    fn test_no_dangling_nodes() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n = i1.add(i2)?;
                n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)
            })?;

            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_dangling_nodes(c.get_main_graph()?, new_g.clone())?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;
            assert!(contexts_deep_equal(&new_c, &c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_dangling_nodes() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let _dangling1 = i1.multiply(i2.clone());
                _dangling1?.set_name("Removed")?;
                let n = i1.add(i2)?;
                n.set_name("Left")?;
                let _dangling2 =
                    n.multiply(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)?;
                n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)
            })?;

            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_dangling_nodes(c.get_main_graph()?, new_g.clone())?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;

            assert!(!contexts_deep_equal(&new_c, &c));
            assert_eq!(c.get_main_graph()?.get_nodes().len(), 8);
            assert_eq!(new_c.get_main_graph()?.get_nodes().len(), 5);
            // Check names
            let new_dangling1 = new_c.retrieve_node(new_g.clone(), "Removed");
            assert!(new_dangling1.is_err());
            let new_n = new_c.retrieve_node(new_g, "Left");
            assert!(new_n.is_ok());
            assert_eq!(new_n?.get_operation(), Operation::Add);
            Ok(())
        }()
        .unwrap();
    }
}
