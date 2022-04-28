use crate::data_types::{scalar_type, Type, UINT64};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Graph, Node};
use crate::inline::data_structures::{log_depth_sum, CombineOp};
use crate::inline::inline_common::{
    pick_prefix_sum_algorithm, DepthOptimizationLevel, InlineState,
};

pub(super) fn inline_iterate_associative(
    graph: Graph,
    initial_state: Node,
    inputs_node: Node,
    optimization_level: DepthOptimizationLevel,
    inliner: &mut dyn InlineState,
) -> Result<(Node, Vec<Node>)> {
    let graph_output_type = graph.get_output_node()?.get_type()?;
    let output_element_type = match graph_output_type {
        Type::Tuple(tuple_types) => (*tuple_types[1]).clone(),
        _ => {
            panic!("Inconsistency with type checker for Iterate output.");
        }
    };
    let inputs_len = match inputs_node.get_type()? {
        Type::Vector(len, _) => len,
        _ => {
            panic!("Inconsistency with type checker");
        }
    };
    if inputs_len == 0 {
        return Ok((initial_state, vec![]));
    }

    let empty_output = match output_element_type {
        Type::Tuple(tuple_types) => tuple_types.is_empty(),
        _ => false,
    };
    let mut inputs = vec![initial_state];
    for i in 0..inputs_len {
        let current_input = inputs_node.vector_get(
            inliner
                .output_graph()
                .constant(scalar_type(UINT64), Value::from_scalar(i, UINT64)?)?,
        )?;
        inputs.push(current_input.clone());
    }
    if inputs[0].get_type()? != inputs[1].get_type()? {
        return Err(runtime_error!(
            "Associative optimization requires state and inputs of the same type"
        ));
    }
    let mut combiner = StateCombiner {
        graph: graph.clone(),
        inliner,
    };
    // For empty outputs, we can be more efficient.
    if empty_output {
        // Outputs for this case are trivial.
        let mut outputs = vec![];
        let empty_tuple = combiner.inliner.output_graph().create_tuple(vec![])?;
        for _ in 0..inputs_len {
            outputs.push(empty_tuple.clone());
        }
        // Compute the final state with logarithmic depth.
        let result = log_depth_sum(&inputs, &mut combiner)?;
        Ok((result, outputs))
    } else {
        let prefix_sums =
            pick_prefix_sum_algorithm(inputs_len, optimization_level)(&inputs, &mut combiner)?;
        let mut outputs = vec![];
        for i in 0..inputs_len {
            inliner.assign_input_nodes(
                graph.clone(),
                vec![
                    prefix_sums[i as usize].clone(),
                    inputs[(i + 1) as usize].clone(),
                ],
            )?;
            let output = inliner.recursively_inline_graph(graph.clone())?;
            inliner.unassign_nodes(graph.clone())?;
            outputs.push(output.tuple_get(1)?);
        }
        Ok((prefix_sums[prefix_sums.len() - 1].clone(), outputs))
    }
}

struct StateCombiner<'a> {
    graph: Graph,
    inliner: &'a mut dyn InlineState,
}

impl<'a> CombineOp<Node> for StateCombiner<'a> {
    fn combine(&mut self, arg1: Node, arg2: Node) -> Result<Node> {
        self.inliner
            .assign_input_nodes(self.graph.clone(), vec![arg1, arg2])?;
        let output = self.inliner.recursively_inline_graph(self.graph.clone())?;
        self.inliner.unassign_nodes(self.graph.clone())?;
        output.tuple_get(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::BIT;
    use crate::graphs::create_context;
    use crate::inline::inline_test_utils::{build_test_data, resolve_tuple_get, MockInlineState};

    #[test]
    fn test_associative_iterate_empty_output() {
        || -> Result<()> {
            let c = create_context()?;
            let (g, initial_state, inputs_node, _input_vals) = build_test_data(c.clone(), UINT64)?;
            let mut inliner = MockInlineState {
                fake_graph: g.clone(),
                inputs: vec![],
                inline_graph_calls: vec![],
                returned_nodes: vec![],
            };
            let g_inline = c.create_graph()?;
            let empty = g_inline.create_tuple(vec![])?;
            g_inline.set_output_node(g_inline.create_tuple(vec![empty.clone(), empty.clone()])?)?;
            let res = inline_iterate_associative(
                g_inline.clone(),
                initial_state.clone(),
                inputs_node.clone(),
                DepthOptimizationLevel::Extreme,
                &mut inliner,
            )?;
            assert_eq!(inliner.inputs.len(), 5);
            assert_eq!(inliner.inline_graph_calls.len(), 5);
            assert_eq!(inliner.returned_nodes.len(), 5);
            // We have 5 elements + initial state (0), so the edges of the tree should be:
            // 0-1, 2-3, 4-5, (0+1)-(2+3), (0+1+2+3)-(4+5).
            assert!(initial_state.clone() == inliner.inputs[0][0]);
            assert!(
                inliner.returned_nodes[0][0] == resolve_tuple_get(inliner.inputs[3][0].clone())
            );
            assert!(
                inliner.returned_nodes[1][0] == resolve_tuple_get(inliner.inputs[3][1].clone())
            );
            assert!(
                inliner.returned_nodes[2][0] == resolve_tuple_get(inliner.inputs[4][1].clone())
            );
            assert!(
                inliner.returned_nodes[3][0] == resolve_tuple_get(inliner.inputs[4][0].clone())
            );
            assert!(inliner.returned_nodes[4][0] == resolve_tuple_get(res.0));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_associative_iterate_empty_input() {
        || -> Result<()> {
            let c = create_context()?;
            let g1 = c.create_graph()?;
            let i1 = g1.input(scalar_type(BIT))?;
            let o1 = g1.create_tuple(vec![i1.clone(), i1.clone()])?;
            o1.set_as_output()?;
            g1.finalize()?;
            g1.set_as_main()?;
            c.finalize()?;
            let output_c = create_context()?;
            let output_g = output_c.create_graph()?;
            let vec = output_g.create_vector(scalar_type(BIT), vec![])?;
            let s0 = output_g.input(scalar_type(BIT))?;
            let mut inliner = MockInlineState {
                fake_graph: output_g.clone(),
                inputs: vec![],
                inline_graph_calls: vec![],
                returned_nodes: vec![],
            };
            let res = inline_iterate_associative(
                g1.clone(),
                s0.clone(),
                vec.clone(),
                DepthOptimizationLevel::Extreme,
                &mut inliner,
            )?;
            assert!(res.1.is_empty());
            assert!(inliner.inline_graph_calls.is_empty());
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_associative_iterate_nonempty_output_min_depth() {
        || -> Result<()> {
            let c = create_context()?;
            let (g, initial_state, inputs_node, _input_vals) = build_test_data(c.clone(), UINT64)?;
            let mut inliner = MockInlineState {
                fake_graph: g.clone(),
                inputs: vec![],
                inline_graph_calls: vec![],
                returned_nodes: vec![],
            };
            let g_inline = c.create_graph()?;
            let inp = g_inline.input(scalar_type(BIT))?;
            g_inline
                .create_tuple(vec![inp.clone(), inp.clone()])?
                .set_as_output()?;
            inline_iterate_associative(
                g_inline.clone(),
                initial_state.clone(),
                inputs_node.clone(),
                DepthOptimizationLevel::Extreme,
                &mut inliner,
            )?;
            assert_eq!(inliner.inputs.len(), 6 + 5 + 5);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_associative_iterate_nonempty_output_max_depth() {
        || -> Result<()> {
            let c = create_context()?;
            let (g, initial_state, inputs_node, _input_vals) = build_test_data(c.clone(), UINT64)?;
            let mut inliner = MockInlineState {
                fake_graph: g.clone(),
                inputs: vec![],
                inline_graph_calls: vec![],
                returned_nodes: vec![],
            };
            let g_inline = c.create_graph()?;
            let inp = g_inline.input(scalar_type(BIT))?;
            g_inline
                .create_tuple(vec![inp.clone(), inp.clone()])?
                .set_as_output()?;
            inline_iterate_associative(
                g_inline.clone(),
                initial_state.clone(),
                inputs_node.clone(),
                DepthOptimizationLevel::Default,
                &mut inliner,
            )?;
            assert_eq!(inliner.inputs.len(), 6 + 1 + 5);
            Ok(())
        }()
        .unwrap();
    }
}
