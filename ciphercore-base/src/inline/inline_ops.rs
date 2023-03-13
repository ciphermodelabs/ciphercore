use crate::custom_ops::ContextMappings;
use crate::data_types::Type;
use crate::errors::Result;
use crate::graphs::{copy_node_name, create_context, Context};
use crate::graphs::{Graph, GraphAnnotation, Node, Operation};
use crate::inline::associative_iterate_inliner::inline_iterate_associative;
use crate::inline::empty_state_iterate_inliner::inline_iterate_empty_state;
use crate::inline::exponential_inliner::inline_iterate_small_state;
pub use crate::inline::inline_common::DepthOptimizationLevel;
use crate::inline::inline_common::InlineState;
use crate::inline::simple_iterate_inliner::inline_iterate_simple;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InlineMode {
    Noop,
    Simple,
    // Parameter is optimization level, the higher - the more we trade performance for depth.
    DepthOptimized(DepthOptimizationLevel),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct InlineConfig {
    pub default_mode: InlineMode,
    pub override_call_mode: Option<InlineMode>,
    pub override_iterate_mode: Option<InlineMode>,
}

impl Default for InlineConfig {
    fn default() -> InlineConfig {
        InlineConfig {
            default_mode: InlineMode::Noop,
            override_call_mode: None,
            override_iterate_mode: None,
        }
    }
}

/// Resolves the inlining mode for the given node (potentially handling things
/// like associative operations in Iterate, or per-node/per-operation overrides).
fn get_mode_for_node(node: Node, config: InlineConfig) -> InlineMode {
    match node.get_operation() {
        Operation::Call => {
            if let Some(mode) = config.override_call_mode {
                return mode;
            }
        }
        Operation::Iterate => {
            if let Some(mode) = config.override_iterate_mode {
                return mode;
            }
        }
        _ => {
            return InlineMode::Noop;
        }
    }
    config.default_mode
}

struct InliningContext {
    config: InlineConfig,
    // Original node -> new node.
    context_mapping: ContextMappings,
    // Node of an inlined subgraph -> new node.
    // The values for the same node can be different at different points of
    // time, since each graph can be inlined more than once.
    // Invariants:
    // -- if a node is present in both context mappings, the one in context_mapping
    //    is "real" (not inlined);
    // -- if a node is present only in context_mapping, it can be both "real" and inlined;
    // -- if it is present only in ephemeral_context_mapping, it cannot be "real".
    ephemeral_context_mapping: ContextMappings,
}

impl InliningContext {
    fn contains_graph(&self, graph: Graph) -> bool {
        self.context_mapping.contains_graph(graph)
    }

    fn get_graph(&self, graph: Graph) -> Graph {
        self.context_mapping.get_graph(graph)
    }

    fn insert_graph(&mut self, old_graph: Graph, new_graph: Graph) {
        self.context_mapping.insert_graph(old_graph, new_graph)
    }

    fn get_node(&self, node: Node) -> Node {
        if self.ephemeral_context_mapping.contains_node(node.clone()) {
            return self.ephemeral_context_mapping.get_node(node);
        }
        self.context_mapping.get_node(node)
    }

    fn insert_node(&mut self, old_node: Node, new_node: Node) {
        if self.context_mapping.contains_node(old_node.clone()) {
            self.ephemeral_context_mapping
                .insert_node(old_node, new_node)
        } else {
            self.context_mapping.insert_node(old_node, new_node)
        }
    }

    fn remove_ephemeral_node(&mut self, old_node: Node) {
        self.ephemeral_context_mapping.remove_node(old_node)
    }
}

struct InlineStateImpl<'a> {
    output_graph: Graph,
    inlining_context: &'a mut InliningContext,
}

impl<'a> InlineState for InlineStateImpl<'a> {
    fn assign_input_nodes(&mut self, graph: Graph, nodes: Vec<Node>) -> Result<()> {
        assign_input_nodes(graph, nodes, self.inlining_context)
    }

    fn unassign_nodes(&mut self, graph: Graph) -> Result<()> {
        unassign_nodes(graph, self.inlining_context)
    }

    fn recursively_inline_graph(&mut self, graph: Graph) -> Result<Node> {
        recursively_inline_graph(graph, self.output_graph.clone(), self.inlining_context)
    }

    fn output_graph(&self) -> Graph {
        self.output_graph.clone()
    }
}

/// Creates a new context from the given context, recursively inlining
/// operations in the main graph with accordance to the given InlineConfig.
/// In case of inlined operations with graph dependencies, these graphs are
/// recursively processed with the same config.
/// The inlining process preserves node annotations and names of nodes in the main graph.
/// The name of a to-be-inlined Call/Iterate node is passed to a node containing its output.
pub fn inline_operations(context: Context, config: InlineConfig) -> Result<Context> {
    context.check_finalized()?;
    // First, collect all graphs reachable from the main graph which won't be
    // inlined.
    let mut graph_ids_to_process = HashSet::<u64>::new();
    let mut graph_ids_seen = HashSet::<u64>::new();
    let main_graph = context.get_main_graph()?;
    graph_ids_to_process.insert(main_graph.get_id());
    graph_ids_seen.insert(main_graph.get_id());
    collect_graphs(
        main_graph.clone(),
        config.clone(),
        &mut graph_ids_to_process,
        &mut graph_ids_seen,
    )?;
    let output_context = create_context()?;
    let mut inlining_context = InliningContext {
        config,
        context_mapping: ContextMappings::default(),
        ephemeral_context_mapping: ContextMappings::default(),
    };
    // Now, apply inlining to all of the collected graphs in the topological order.
    for graph in context.get_graphs() {
        if graph_ids_to_process.contains(&graph.get_id()) {
            let new_graph = output_context.create_graph()?;
            for annotation in graph.get_annotations()? {
                new_graph.add_annotation(annotation)?;
            }
            inlining_context.insert_graph(graph.clone(), new_graph.clone());
            let output_node =
                recursively_inline_graph(graph, new_graph.clone(), &mut inlining_context)?;
            new_graph.set_output_node(output_node)?;
            new_graph.finalize()?;
        }
    }

    output_context.set_main_graph(inlining_context.get_graph(main_graph))?;
    output_context.finalize()?;
    Ok(output_context)
}

/// Determines all subgraphs reachable from graph which won't be inlined.
fn collect_graphs(
    graph: Graph,
    config: InlineConfig,
    graph_ids_to_process: &mut HashSet<u64>,
    graph_ids_seen: &mut HashSet<u64>,
) -> Result<()> {
    graph.check_finalized()?;
    for node in graph.get_nodes() {
        let mode = get_mode_for_node(node.clone(), config.clone());
        for dependency in node.get_graph_dependencies() {
            if mode == InlineMode::Noop {
                // It is possible that the same graph is inlined in one place and is not inlined in another.
                graph_ids_to_process.insert(dependency.get_id());
            }
            if !graph_ids_seen.contains(&dependency.get_id()) {
                graph_ids_seen.insert(dependency.get_id());
                collect_graphs(
                    dependency,
                    config.clone(),
                    graph_ids_to_process,
                    graph_ids_seen,
                )?;
            }
        }
    }
    Ok(())
}

fn recursively_inline_graph(
    graph: Graph,
    output_graph: Graph,
    inlining_context: &mut InliningContext,
) -> Result<Node> {
    graph.check_finalized()?;
    if output_graph.is_finalized() {
        return Err(runtime_error!("Cannot modify finalized graph"));
    }
    let is_main_graph = graph.get_context().get_main_graph()? == graph;
    for node in graph.get_nodes() {
        // The node can be in the mapping either if we are in a recursive call and processing a subgraph
        // with pre-defined inputs, or if we're inlining graph which was called but not inlined
        // in another place. The former can only happen with ephemeral_context_mapping.
        if inlining_context
            .ephemeral_context_mapping
            .contains_node(node.clone())
        {
            if !matches!(node.get_operation(), Operation::Input(_)) {
                panic!("Logic error: non-input node is already processed");
            }
            continue;
        }
        let mut new_dependencies = vec![];
        for node_dep in node.get_node_dependencies() {
            new_dependencies.push(inlining_context.get_node(node_dep));
        }
        let mode = get_mode_for_node(node.clone(), inlining_context.config.clone());
        if mode == InlineMode::Noop {
            let mut new_graph_dependencies = vec![];
            for subgraph in node.get_graph_dependencies() {
                if !inlining_context.contains_graph(subgraph.clone()) {
                    panic!("Logic error: all reachable subgraphs should be processed");
                }
                new_graph_dependencies.push(inlining_context.get_graph(subgraph).clone());
            }
            let new_node = output_graph.add_node(
                new_dependencies,
                new_graph_dependencies,
                node.get_operation(),
            )?;
            inlining_context.insert_node(node.clone(), new_node.clone());
            let annotations = node.get_annotations()?;
            if !annotations.is_empty() {
                for annotation in annotations {
                    new_node.add_annotation(annotation)?;
                }
            }

            // Every node name in the main graph is copied
            if is_main_graph {
                copy_node_name(node, new_node)?;
            }
            continue;
        }
        match node.get_operation() {
            Operation::Call => {
                let output_node = inline_call(
                    node.get_graph_dependencies()[0].clone(),
                    output_graph.clone(),
                    new_dependencies,
                    mode,
                    inlining_context,
                )?;
                // Every node name in the main graph is copied
                if is_main_graph {
                    copy_node_name(node.clone(), output_node.clone())?;
                }
                inlining_context.insert_node(node.clone(), output_node.clone());
            }
            Operation::Iterate => {
                let output_node = inline_iterate(
                    node.get_graph_dependencies()[0].clone(),
                    output_graph.clone(),
                    new_dependencies,
                    mode,
                    inlining_context,
                )?;
                // Every node name in the main graph is copied
                if is_main_graph {
                    copy_node_name(node.clone(), output_node.clone())?;
                }
                inlining_context.insert_node(node.clone(), output_node.clone());
            }
            _ => {
                return Err(runtime_error!(
                    "Inlining is not implemented for the operation"
                ));
            }
        }
    }
    Ok(inlining_context.get_node(graph.get_output_node()?))
}

fn assign_input_nodes(
    graph: Graph,
    nodes: Vec<Node>,
    inlining_context: &mut InliningContext,
) -> Result<()> {
    let mut input_nodes = vec![];
    for node in graph.get_nodes() {
        if let Operation::Input(_) = node.get_operation() {
            input_nodes.push(node.clone());
        }
    }
    if input_nodes.len() != nodes.len() {
        return Err(runtime_error!("Mismatch in the number of nodes"));
    }
    for (input_node, node) in input_nodes.iter().zip(nodes.iter()) {
        inlining_context
            .ephemeral_context_mapping
            .insert_node(input_node.clone(), node.clone());
    }
    Ok(())
}

fn unassign_nodes(graph: Graph, inlining_context: &mut InliningContext) -> Result<()> {
    for node in graph.get_nodes() {
        // The context mapping can be missing the node if it is not reachable
        // from the output node.
        if inlining_context
            .ephemeral_context_mapping
            .contains_node(node.clone())
        {
            inlining_context.remove_ephemeral_node(node);
        }
    }
    Ok(())
}

fn inline_call(
    graph: Graph,
    output_graph: Graph,
    dependencies: Vec<Node>,
    mode: InlineMode,
    inlining_context: &mut InliningContext,
) -> Result<Node> {
    match mode {
        InlineMode::Simple | InlineMode::DepthOptimized(_) => {
            assign_input_nodes(graph.clone(), dependencies, inlining_context)?;
            let result = recursively_inline_graph(graph.clone(), output_graph, inlining_context)?;
            unassign_nodes(graph, inlining_context)?;
            Ok(result)
        }
        _ => Err(runtime_error!(
            "Optimization mode is not implemented for Call"
        )),
    }
}

fn is_empty_tuple(value_type: Type) -> bool {
    match value_type {
        Type::Tuple(inner_types) => inner_types.is_empty(),
        _ => false,
    }
}

fn inline_iterate(
    graph: Graph,
    output_graph: Graph,
    dependencies: Vec<Node>,
    mode: InlineMode,
    inlining_context: &mut InliningContext,
) -> Result<Node> {
    let inputs_node = dependencies[1].clone();
    let initial_state = dependencies[0].clone();
    let graph_output_type = graph.get_output_node()?.get_type()?;
    let output_element_type = match graph_output_type {
        Type::Tuple(tuple_types) => (*tuple_types[1]).clone(),
        _ => {
            panic!("Inconsistency with type checker for Iterate output.");
        }
    };

    let mut inline_state = InlineStateImpl {
        output_graph: output_graph.clone(),
        inlining_context,
    };

    let mut simple_inliner = || -> Result<(Node, Vec<Node>)> {
        inline_iterate_simple(
            graph.clone(),
            initial_state.clone(),
            inputs_node.clone(),
            &mut inline_state,
        )
    };

    let (final_state, outputs) = match mode {
        InlineMode::Simple => simple_inliner(),
        InlineMode::DepthOptimized(optimization_level) => {
            if is_empty_tuple(initial_state.get_type()?) {
                inline_iterate_empty_state(
                    graph.clone(),
                    initial_state.clone(),
                    inputs_node.clone(),
                    &mut inline_state,
                )
            } else if graph
                .get_annotations()?
                .contains(&GraphAnnotation::AssociativeOperation)
            {
                inline_iterate_associative(
                    graph.clone(),
                    initial_state.clone(),
                    inputs_node.clone(),
                    optimization_level,
                    &mut inline_state,
                )
            } else if graph
                .get_annotations()?
                .contains(&GraphAnnotation::OneBitState)
            {
                inline_iterate_small_state(
                    true,
                    optimization_level,
                    graph.clone(),
                    initial_state.clone(),
                    inputs_node.clone(),
                    &mut inline_state,
                )
            } else if graph
                .get_annotations()?
                .contains(&GraphAnnotation::SmallState)
            {
                inline_iterate_small_state(
                    false,
                    optimization_level,
                    graph.clone(),
                    initial_state.clone(),
                    inputs_node.clone(),
                    &mut inline_state,
                )
            } else {
                simple_inliner()
            }
        }
        _ => Err(runtime_error!(
            "Optimization mode is not implemented for Iterate"
        )),
    }?;
    let final_result = output_graph.create_vector(output_element_type, outputs)?;
    output_graph.create_tuple(vec![final_state, final_result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::{array_type, scalar_type, tuple_type, BIT, UINT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::graphs::util::simple_context;
    use crate::graphs::{contexts_deep_equal, NodeAnnotation, SliceElement};
    use crate::ops::comparisons::{Equal, LessThan};
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    #[test]
    fn test_noop() {
        || -> Result<()> {
            let c = create_context()?;
            let g1 = c.create_graph()?;
            let i1 = g1.input(scalar_type(BIT))?;
            g1.set_output_node(i1)?;
            g1.finalize()?;
            let g2 = c.create_graph()?;
            let i2 = g2.input(scalar_type(BIT))?;
            let o2 = g2.call(g1, vec![i2])?;
            o2.set_name("Call g1")?;
            o2.add_annotation(NodeAnnotation::AssociativeOperation)?;
            g2.set_output_node(o2.clone())?;
            g2.finalize()?;
            c.set_main_graph(g2)?;
            c.finalize()?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: InlineMode::Noop,
                    ..Default::default()
                },
            )?;
            assert!(contexts_deep_equal(c.clone(), c_out.clone()));
            assert_eq!(o2.clone().get_annotations()?.len(), 1);
            Ok(())
        }()
        .unwrap();
    }

    fn verify_on_all_inputs(g1: Graph, g2: Graph) -> Result<()> {
        for x in vec![0, 1] {
            for y in vec![0, 1] {
                for z in vec![0, 1] {
                    let inputs = vec![
                        Value::from_scalar(x, BIT)?,
                        Value::from_scalar(y, BIT)?,
                        Value::from_scalar(z, BIT)?,
                    ];
                    let old_result = random_evaluate(g1.clone(), inputs.clone())?;
                    let new_result = random_evaluate(g2.clone(), inputs.clone())?;
                    assert_eq!(old_result, new_result);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_inline_call_simple() {
        || -> Result<()> {
            let c = create_context()?;
            let g1 = c.create_graph()?;
            let i11 = g1.input(scalar_type(BIT))?;
            i11.set_name("First input")?;
            let i12 = g1.input(scalar_type(BIT))?;
            i12.set_name("Second input")?;
            let sum12 = g1.add(i11.clone(), i12.clone())?;
            sum12.set_name("Output of g1")?;
            g1.set_output_node(sum12)?;
            g1.finalize()?;
            let g2 = c.create_graph()?;
            let i21 = g2.input(scalar_type(BIT))?;
            i21.set_name("First input")?;
            let i22 = g2.input(scalar_type(BIT))?;
            let i23 = g2.input(scalar_type(BIT))?;
            let o1 = g2.call(g1, vec![i21, i22])?;
            o1.set_name("Output of calling g1")?;
            let o2 = g2.multiply(o1, i23)?;
            g2.set_output_node(o2.clone())?;
            g2.finalize()?;
            c.set_main_graph(g2.clone())?;
            c.finalize()?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: InlineMode::Simple,
                    ..Default::default()
                },
            )?;
            assert!(!contexts_deep_equal(c.clone(), c_out.clone()));
            assert!(c_out.is_finalized());
            assert_eq!(c_out.get_graphs().len(), 1);
            let g_out = c_out.get_main_graph()?;
            for node in g_out.get_nodes() {
                assert_eq!(node.get_graph_dependencies().len(), 0);
            }
            // Check that the graphs are equivalent as functions.
            verify_on_all_inputs(g2.clone(), g_out.clone())?;
            // Check names
            let inlined_g = c_out.get_main_graph()?;
            let named_call_result = c_out.retrieve_node(inlined_g.clone(), "Output of calling g1");
            assert!(named_call_result.is_ok());
            let named_call = named_call_result?;
            assert_eq!(named_call.get_operation(), Operation::Add);
            let named_input_result = c_out.retrieve_node(inlined_g.clone(), "First input");
            assert!(named_input_result.is_ok());
            let named_input = named_input_result?;
            assert_eq!(
                named_input.get_operation(),
                Operation::Input(scalar_type(BIT))
            );
            let named_input2_result = c_out.retrieve_node(inlined_g, "Second input");
            assert!(named_input2_result.is_err());
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_inline_multiple_calls() {
        || -> Result<()> {
            let c = create_context()?;

            let g0 = c.create_graph()?;
            {
                let i01 = g0.input(scalar_type(BIT))?;
                let i02 = g0.input(scalar_type(BIT))?;
                let sum12 = g0.add(i01.clone(), i02.clone())?;
                g0.set_output_node(sum12)?;
                g0.finalize()?;
            }

            let g1 = c.create_graph()?;
            {
                let i11 = g1.input(scalar_type(BIT))?;
                let i12 = g1.input(scalar_type(BIT))?;
                let sum12 = g1.call(g0, vec![i11, i12])?;
                g1.set_output_node(sum12)?;
                g1.finalize()?;
            }

            let g2 = c.create_graph()?;
            {
                let i21 = g2.input(scalar_type(BIT))?;
                let i22 = g2.input(scalar_type(BIT))?;
                let i23 = g2.input(scalar_type(BIT))?;
                let o1 = g2.call(g1.clone(), vec![i21, i22])?;
                // Call g1 multiple times.
                let o2 = g2.call(g1, vec![o1, i23])?;
                g2.set_output_node(o2.clone())?;
                g2.finalize()?;
            }

            c.set_main_graph(g2.clone())?;
            c.finalize()?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: InlineMode::Simple,
                    ..Default::default()
                },
            )?;
            verify_on_all_inputs(g2.clone(), c_out.get_main_graph()?)?;
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_inline_in_one_place_no_inline_in_another() {
        || -> Result<()> {
            let c = create_context()?;

            let g0 = c.create_graph()?;
            {
                let i01 = g0.input(scalar_type(BIT))?;
                let i02 = g0.input(scalar_type(BIT))?;
                let sum12 = g0.add(i01.clone(), i02.clone())?;
                let out = g0.create_tuple(vec![sum12.clone(), sum12.clone()])?;
                g0.set_output_node(out)?;
                g0.finalize()?;
            }

            let g1 = c.create_graph()?;
            {
                let i21 = g1.input(scalar_type(BIT))?;
                let i22 = g1.input(scalar_type(BIT))?;
                let i23 = g1.input(scalar_type(BIT))?;
                let o1 = g1.tuple_get(g1.call(g0.clone(), vec![i21, i22])?, 0)?;
                let input_vec = g1.repeat(i23, 1)?;
                let o2 = g1.tuple_get(g1.iterate(g0.clone(), o1, input_vec)?, 0)?;
                g1.set_output_node(o2.clone())?;
                g1.finalize()?;
            }
            c.set_main_graph(g1.clone())?;
            c.finalize()?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: InlineMode::Simple,
                    override_iterate_mode: Some(InlineMode::Noop),
                    ..Default::default()
                },
            )?;
            verify_on_all_inputs(g1.clone(), c_out.get_main_graph()?)?;
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_inline_iterate() {
        let helper = |mode| -> Result<()> {
            let c = create_context()?;

            let g0 = c.create_graph()?;
            {
                let i01 = g0.input(scalar_type(BIT))?;
                let i02 = g0.input(scalar_type(BIT))?;
                let sum12 = g0.add(i01.clone(), i02.clone())?;
                let out = g0.create_tuple(vec![sum12.clone(), sum12.clone()])?;
                out.set_name("CreateTuple output")?;
                g0.set_output_node(out)?;
                g0.finalize()?;
            }

            let g1 = c.create_graph()?;
            {
                let i21 = g1.input(scalar_type(BIT))?;
                let i22 = g1.input(scalar_type(BIT))?;
                let i23 = g1.input(scalar_type(BIT))?;
                let o1 = g1.tuple_get(g1.call(g0.clone(), vec![i21, i22])?, 0)?;
                let input_vec = g1.repeat(i23, 5)?;
                let o2 = g1.iterate(g0.clone(), o1, input_vec)?;
                o2.set_name("Iterate output")?;
                let o3 = g1.tuple_get(o2, 0)?;
                g1.set_output_node(o3.clone())?;
                g1.finalize()?;
            }
            c.set_main_graph(g1.clone())?;
            c.finalize()?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: mode,
                    ..Default::default()
                },
            )?;
            assert_eq!(c_out.get_graphs().len(), 1);
            verify_on_all_inputs(g1.clone(), c_out.get_main_graph()?)?;
            // Check names
            let inlined_g = c_out.get_main_graph()?;
            let named_iterate_result = c_out.retrieve_node(inlined_g.clone(), "Iterate output");
            assert!(named_iterate_result.is_ok());
            let named_iterate = named_iterate_result?;
            assert_eq!(named_iterate.get_operation(), Operation::CreateTuple);
            Ok(())
        };
        helper(InlineMode::Simple).unwrap();
        helper(InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme)).unwrap();
    }

    #[test]
    fn test_nested_iterate() {
        let helper = |mode| -> Result<()> {
            let c = create_context()?;

            let g0 = c.create_graph()?;
            {
                let i01 = g0.input(scalar_type(BIT))?;
                let i02 = g0.input(scalar_type(BIT))?;
                let sum12 = g0.add(i01.clone(), i02.clone())?;
                let out = g0.create_tuple(vec![sum12.clone(), sum12.clone()])?;
                g0.set_output_node(out)?;
                g0.finalize()?;
            }
            let g1 = c.create_graph()?;
            {
                let i21 = g1.input(scalar_type(BIT))?;
                let i22 = g1.input(scalar_type(BIT))?;
                let i23 = i21.clone();
                let o1 = g1.tuple_get(g1.call(g0.clone(), vec![i21, i22])?, 0)?;
                let input_vec = g1.repeat(i23, 5)?;
                let o2 = g1.iterate(g0.clone(), o1, input_vec)?;
                g1.set_output_node(o2.clone())?;
                g1.finalize()?;
            }
            let g2 = c.create_graph()?;
            {
                let i21 = g2.input(scalar_type(BIT))?;
                let i22 = g2.input(scalar_type(BIT))?;
                let i23 = g2.input(scalar_type(BIT))?;
                let o1 = g2.tuple_get(g2.call(g0.clone(), vec![i21.clone(), i22.clone()])?, 0)?;
                let input_vec = g2.create_vector(
                    scalar_type(BIT),
                    vec![
                        i21.clone(),
                        i22.clone(),
                        i23.clone(),
                        o1.clone(),
                        o1.clone(),
                        o1.clone(),
                    ],
                )?;
                let o2 = g2.iterate(g1.clone(), o1, input_vec)?;
                g2.set_output_node(o2.clone())?;
                g2.finalize()?;
            }
            c.set_main_graph(g2.clone())?;
            c.finalize()?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: mode,
                    ..Default::default()
                },
            )?;
            assert_eq!(c_out.get_graphs().len(), 1);
            verify_on_all_inputs(g2.clone(), c_out.get_main_graph()?)?;
            Ok(())
        };
        helper(InlineMode::Simple).unwrap();
        helper(InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme)).unwrap();
    }

    fn verify_on_random_inputs(g1: Graph, g2: Graph) -> Result<()> {
        let mut rng = StdRng::seed_from_u64(57);
        for _ in 0..100 {
            let mut inputs = vec![];
            for _ in 0..4 {
                let arr: Vec<u64> = (0..32).map(|_| rng.next_u64() % 2).collect();
                inputs.push(Value::from_flattened_array(&arr, BIT)?);
            }
            let old_result = random_evaluate(g1.clone(), inputs.clone())?;
            let new_result = random_evaluate(g2.clone(), inputs.clone())?;
            assert_eq!(old_result, new_result);
        }
        Ok(())
    }

    enum IterateOutput {
        Empty,
        State,
    }

    fn generate_context_for_associative_iterate(iterate_output: IterateOutput) -> Result<Context> {
        let c = create_context()?;
        let bit_type = array_type(vec![32], BIT);
        let g0 = c.create_graph()?;
        {
            let state = g0.input(bit_type.clone())?;
            let input = g0.input(bit_type.clone())?;
            let output_state = g0.multiply(state, input)?;
            let output = match iterate_output {
                IterateOutput::Empty => {
                    g0.create_tuple(vec![output_state, g0.create_tuple(vec![])?])?
                }
                IterateOutput::State => {
                    g0.create_tuple(vec![output_state.clone(), output_state.clone()])?
                }
            };
            g0.set_output_node(output)?;
            g0.add_annotation(GraphAnnotation::AssociativeOperation)?;
            g0.finalize()?;
        }

        let g1 = c.create_graph()?;
        {
            let i0 = g1.input(bit_type.clone())?;
            let i1 = g1.input(bit_type.clone())?;
            let i2 = g1.input(bit_type.clone())?;
            let i3 = g1.input(bit_type.clone())?;
            let input_vec = g1.create_vector(
                bit_type.clone(),
                vec![i0.clone(), i1.clone(), i2.clone(), i3.clone()],
            )?;
            let iterate_out = g1.iterate(g0.clone(), i0, input_vec)?;
            let output = match iterate_output {
                IterateOutput::State => {
                    let mut output = g1.tuple_get(iterate_out.clone(), 0)?;
                    let vec = g1.tuple_get(iterate_out, 1)?;
                    for i in 0..4 {
                        output = output.add(vec.vector_get(
                            g1.constant(scalar_type(UINT64), Value::from_scalar(i, UINT64)?)?,
                        )?)?;
                    }
                    output
                }
                _ => g1.tuple_get(iterate_out.clone(), 0)?,
            };
            g1.set_output_node(output.clone())?;
            g1.finalize()?;
        }
        c.set_main_graph(g1.clone())?;
        c.finalize()?;
        Ok(c)
    }

    #[test]
    fn test_inline_iterate_associative() {
        let helper = |mode, inline_output| -> Result<()> {
            let c = generate_context_for_associative_iterate(inline_output)?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: mode,
                    ..Default::default()
                },
            )?;
            assert_eq!(c_out.get_graphs().len(), 1);
            verify_on_random_inputs(c.get_main_graph()?, c_out.get_main_graph()?)?;
            Ok(())
        };
        helper(InlineMode::Simple, IterateOutput::Empty).unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            IterateOutput::Empty,
        )
        .unwrap();
        // Non-empty output.
        helper(InlineMode::Simple, IterateOutput::Empty).unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            IterateOutput::State,
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            IterateOutput::State,
        )
        .unwrap();
    }

    #[test]
    fn test_empty_state_iterate() {
        let helper = |mode| -> Result<()> {
            let c = create_context()?;

            let g0 = c.create_graph()?;
            {
                let i01 = g0.input(tuple_type(vec![]))?;
                let i02 = g0.input(scalar_type(BIT))?;
                let out = g0.create_tuple(vec![i01.clone(), i02.clone()])?;
                g0.set_output_node(out)?;
                g0.finalize()?;
            }

            let g1 = c.create_graph()?;
            {
                let i21 = g1.input(scalar_type(BIT))?;
                let i22 = g1.input(scalar_type(BIT))?;
                let i23 = g1.input(scalar_type(BIT))?;
                let input_vec =
                    g1.create_vector(i21.get_type()?, vec![i21.clone(), i22.clone(), i23.clone()])?;
                let o1 = g1.tuple_get(
                    g1.iterate(g0.clone(), g1.create_tuple(vec![])?, input_vec)?,
                    1,
                )?;
                let o2 = g1.add(
                    g1.vector_get(
                        o1.clone(),
                        g1.constant(scalar_type(UINT64), Value::from_scalar(0, UINT64)?)?,
                    )?,
                    g1.vector_get(
                        o1.clone(),
                        g1.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?,
                    )?,
                )?;
                g1.set_output_node(o2.clone())?;
                g1.finalize()?;
            }
            c.set_main_graph(g1.clone())?;
            c.finalize()?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: mode,
                    ..Default::default()
                },
            )?;
            assert_eq!(c_out.get_graphs().len(), 1);
            verify_on_all_inputs(g1.clone(), c_out.get_main_graph()?)?;
            Ok(())
        };
        helper(InlineMode::Simple).unwrap();
        helper(InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme)).unwrap();
    }

    fn generate_context_for_small_state(
        single_bit: bool,
        nonempty_output: bool,
        dims: Vec<u64>,
    ) -> Result<Context> {
        let c = create_context()?;
        let bit_type = array_type(dims, BIT);
        let g0 = c.create_graph()?;
        {
            if single_bit {
                let state = g0.input(bit_type.clone())?;
                let input = g0.input(bit_type.clone())?;
                let output_state = g0.multiply(state, input)?;
                let output_val = if nonempty_output {
                    output_state.clone()
                } else {
                    g0.create_tuple(vec![])?
                };
                let output = g0.create_tuple(vec![output_state, output_val])?;
                g0.set_output_node(output)?;
                g0.add_annotation(GraphAnnotation::OneBitState)?;
                g0.finalize()?;
            } else {
                let state = g0.input(bit_type.clone())?;
                let input = g0.input(bit_type.clone())?;
                let tmp_state = g0.add(state, input)?;
                let shape = tmp_state.get_type()?.get_shape();
                let mut bit_columns = vec![];
                for bit_index in 0..shape[shape.len() - 1] {
                    bit_columns.push(tmp_state.get_slice(vec![
                        SliceElement::Ellipsis,
                        SliceElement::SingleIndex(bit_index as i64),
                    ])?);
                }
                for bit_index in 1..shape[shape.len() - 1] {
                    bit_columns[bit_index as usize] = bit_columns[bit_index as usize]
                        .multiply(bit_columns[bit_index as usize - 1].clone())?;
                }
                let mut output_state = g0
                    .create_vector(bit_columns[0].get_type()?, bit_columns)?
                    .vector_to_array()?;
                let mut permutation: Vec<u64> = (0..shape.len()).map(|x| x as u64).collect();
                permutation.rotate_left(1);
                output_state = output_state.permute_axes(permutation)?;

                let output_val = if nonempty_output {
                    output_state.clone()
                } else {
                    g0.create_tuple(vec![])?
                };
                let output = g0.create_tuple(vec![output_state, output_val])?;
                g0.set_output_node(output)?;
                g0.add_annotation(GraphAnnotation::SmallState)?;
                g0.finalize()?;
            }
        }

        let g1 = c.create_graph()?;
        {
            let i0 = g1.input(bit_type.clone())?;
            let i1 = g1.input(bit_type.clone())?;
            let i2 = g1.input(bit_type.clone())?;
            let i3 = g1.input(bit_type.clone())?;
            let input_vec = g1.create_vector(
                bit_type.clone(),
                vec![i0.clone(), i1.clone(), i2.clone(), i3.clone()],
            )?;
            let iterate_out = g1.iterate(g0.clone(), i0, input_vec)?;
            let output = if nonempty_output {
                let mut total = g1.tuple_get(iterate_out.clone(), 0)?;
                let vec = g1.tuple_get(iterate_out, 1)?;
                for i in 0..4 {
                    total = total.add(vec.vector_get(
                        g1.constant(scalar_type(UINT64), Value::from_scalar(i, UINT64)?)?,
                    )?)?;
                }
                total
            } else {
                g1.tuple_get(iterate_out, 0)?
            };
            g1.set_output_node(output.clone())?;
            g1.finalize()?;
        }
        c.set_main_graph(g1.clone())?;
        c.finalize()?;
        Ok(c)
    }

    #[test]
    fn test_small_state_iterate() {
        let helper = |mode, single_bit, shape| -> Result<()> {
            let c = generate_context_for_small_state(single_bit, false, shape)?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: mode,
                    ..Default::default()
                },
            )?;
            assert_eq!(c_out.get_graphs().len(), 1);
            verify_on_random_inputs(c.get_main_graph()?, c_out.get_main_graph()?)?;
            Ok(())
        };
        helper(InlineMode::Simple, true, vec![32]).unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            true,
            vec![32],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            true,
            vec![32, 1],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            false,
            vec![32, 1],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            false,
            vec![16, 2],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            false,
            vec![4, 4, 2],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            true,
            vec![4, 4, 2],
        )
        .unwrap();
        assert!(helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            false,
            vec![4, 1, 8]
        )
        .is_err());
        // TODO: also a test with a scalar.
    }

    #[test]
    fn test_small_state_iterate_nonempty_output() {
        let helper = |mode, single_bit, shape| -> Result<()> {
            let c = generate_context_for_small_state(single_bit, true, shape)?;
            let c_out = inline_operations(
                c.clone(),
                InlineConfig {
                    default_mode: mode,
                    ..Default::default()
                },
            )?;
            assert_eq!(c_out.get_graphs().len(), 1);
            verify_on_random_inputs(c.get_main_graph()?, c_out.get_main_graph()?)?;
            Ok(())
        };
        helper(InlineMode::Simple, true, vec![32]).unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            true,
            vec![32],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            true,
            vec![32, 1],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            false,
            vec![32, 1],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            false,
            vec![16, 2],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            false,
            vec![4, 4, 2],
        )
        .unwrap();
        helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            true,
            vec![4, 4, 2],
        )
        .unwrap();
        assert!(helper(
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme),
            false,
            vec![4, 1, 8]
        )
        .is_err());
        // TODO: also a test with a scalar.
    }

    #[test]
    fn test_small_state_iterate_comparisons() {
        let helper = |equal: bool, mode, x1: Vec<u64>, x2: Vec<u64>| -> Result<Vec<u64>> {
            let c = simple_context(|g| {
                let i1 = g.input(array_type(vec![x1.len() as u64], UINT64))?;
                let i2 = g.input(array_type(vec![x2.len() as u64], UINT64))?;
                if equal {
                    g.custom_op(CustomOperation::new(Equal {}), vec![i1.a2b()?, i2.a2b()?])
                } else {
                    g.custom_op(
                        CustomOperation::new(LessThan {
                            signed_comparison: false,
                        }),
                        vec![i1.a2b()?, i2.a2b()?],
                    )
                }
            })?;
            let mapped_c = run_instantiation_pass(c)?.get_context();
            let c_out = inline_operations(
                mapped_c.clone(),
                InlineConfig {
                    default_mode: mode,
                    ..Default::default()
                },
            )?;
            let inputs = vec![
                Value::from_flattened_array(&x1, UINT64)?,
                Value::from_flattened_array(&x2, UINT64)?,
            ];
            let result = random_evaluate(c_out.get_main_graph()?.clone(), inputs.clone())?;
            result.to_flattened_array_u64(array_type(vec![x1.len() as u64], BIT))
        };
        let res_less_than = helper(
            false,
            InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            vec![1, 2, 3, 4, 5],
            vec![5, 4, 3, 2, 1],
        )
        .unwrap();
        assert_eq!(res_less_than, vec![1, 1, 0, 0, 0]);
        let res_equal = helper(
            true,
            InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            vec![1, 2, 3, 4, 5],
            vec![5, 4, 3, 2, 1],
        )
        .unwrap();
        assert_eq!(res_equal, vec![0, 0, 1, 0, 0]);
    }
}
