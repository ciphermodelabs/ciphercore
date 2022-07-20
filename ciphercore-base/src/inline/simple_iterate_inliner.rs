use crate::data_types::{Type, UINT64};
use crate::errors::Result;
use crate::graphs::{Graph, Node};
use crate::inline::inline_common::InlineState;
use crate::ops::utils::constant_scalar;

pub(super) fn inline_iterate_simple(
    graph: Graph,
    initial_state: Node,
    inputs_node: Node,
    inliner: &mut dyn InlineState,
) -> Result<(Node, Vec<Node>)> {
    let mut state = initial_state;
    let mut outputs = vec![];
    let inputs_len = match inputs_node.get_type()? {
        Type::Vector(len, _) => len,
        _ => {
            panic!("Inconsistency with type checker");
        }
    };
    for i in 0..inputs_len {
        let current_input =
            inputs_node.vector_get(constant_scalar(&inliner.output_graph(), i, UINT64)?)?;
        inliner.assign_input_nodes(graph.clone(), vec![state.clone(), current_input.clone()])?;
        let result = inliner.recursively_inline_graph(graph.clone())?;
        inliner.unassign_nodes(graph.clone())?;
        state = result.tuple_get(0)?.clone();
        outputs.push(result.tuple_get(1)?.clone());
    }
    Ok((state, outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::create_context;
    use crate::inline::inline_test_utils::{build_test_data, resolve_tuple_get, MockInlineState};

    #[test]
    fn test_simple_iterate() {
        || -> Result<()> {
            let c = create_context()?;
            let (g, initial_state, inputs_node, input_vals) = build_test_data(c.clone(), UINT64)?;
            let mut inliner = MockInlineState {
                fake_graph: g.clone(),
                inputs: vec![],
                inline_graph_calls: vec![],
                returned_nodes: vec![],
            };
            let res = inline_iterate_simple(
                g.clone(),
                initial_state.clone(),
                inputs_node.clone(),
                &mut inliner,
            )?;
            assert_eq!(inliner.inputs.len(), 5);
            assert_eq!(inliner.inline_graph_calls.len(), 5);
            assert_eq!(inliner.returned_nodes.len(), 5);
            assert!(initial_state.clone() == inliner.inputs[0][0]);
            assert!(resolve_tuple_get(res.0) == inliner.returned_nodes[4][0]);
            for i in 0..input_vals.len() {
                assert!(resolve_tuple_get(res.1[i].clone()) == inliner.returned_nodes[i][1]);
            }
            for i in 1..input_vals.len() {
                assert!(
                    inliner.returned_nodes[i - 1][0]
                        == resolve_tuple_get(inliner.inputs[i][0].clone())
                );
            }
            Ok(())
        }()
        .unwrap();
    }
}
