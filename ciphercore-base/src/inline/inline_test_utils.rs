use crate::data_types::{scalar_type, ScalarType, BIT};
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, Operation};
use crate::inline::inline_common::InlineState;
use crate::ops::utils::constant_scalar;

pub(super) struct MockInlineState {
    pub fake_graph: Graph,
    pub inputs: Vec<Vec<Node>>,
    pub inline_graph_calls: Vec<Graph>,
    pub returned_nodes: Vec<Vec<Node>>,
}

impl InlineState for MockInlineState {
    fn assign_input_nodes(&mut self, _graph: Graph, nodes: Vec<Node>) -> Result<()> {
        self.inputs.push(nodes.clone());
        Ok(())
    }

    fn unassign_nodes(&mut self, _graph: Graph) -> Result<()> {
        Ok(())
    }

    fn recursively_inline_graph(&mut self, graph: Graph) -> Result<Node> {
        self.inline_graph_calls.push(graph.clone());
        let nodes = vec![
            constant_scalar(&self.fake_graph, 0, BIT)?,
            self.fake_graph.create_tuple(vec![])?,
        ];
        self.returned_nodes.push(nodes.clone());
        self.fake_graph.create_tuple(nodes)
    }

    fn output_graph(&self) -> Graph {
        self.fake_graph.clone()
    }
}

pub(super) fn build_test_data(
    c: Context,
    st: ScalarType,
) -> Result<(Graph, Node, Node, Vec<Node>)> {
    let g = c.create_graph()?;
    let initial_state = constant_scalar(&g, if st == BIT { 1 } else { 42 }, st)?;
    let input_vals = if st == BIT {
        vec![1, 0, 0, 1, 1]
    } else {
        vec![1, 2, 3, 42, 57]
    };
    let mut inputs = vec![];
    for i in input_vals {
        let val = constant_scalar(&g, i, st)?;
        inputs.push(val.clone());
    }
    let inputs_node = g.create_vector(scalar_type(st), inputs.clone())?;
    Ok((
        g.clone(),
        initial_state.clone(),
        inputs_node.clone(),
        inputs,
    ))
}

pub(super) fn resolve_tuple_get(node: Node) -> Node {
    if let Operation::TupleGet(index) = node.get_operation() {
        let tuple = node.get_node_dependencies()[0].clone();
        // We assume that this is an output of CreateTuple, which can be wrong in general.
        let elements = tuple.get_node_dependencies();
        return elements[index as usize].clone();
    }
    return node.clone();
}
