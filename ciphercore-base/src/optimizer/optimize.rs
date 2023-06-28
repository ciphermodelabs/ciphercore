use crate::custom_ops::{ContextMappings, MappedContext};
use crate::errors::Result;
use crate::evaluators::Evaluator;
use crate::graphs::{create_context, Context, Graph, Operation};
use crate::optimizer::constant_optimizer::optimize_graph_constants;
use crate::optimizer::dangling_nodes_optimizer::optimize_graph_dangling_nodes;
use crate::optimizer::meta_operation_optimizer::optimize_graph_meta_operations;
use crate::random::PRNG;

use super::duplicates_optimizer::optimize_graph_duplicates;

/// Applies common optimizations to all graphs in the context.
/// The graphs must be fully inlined.
/// The primary targets of the optimizations here are to remove inefficiencies
/// which happen because of the boilerplate from Iterate inlining.
pub fn optimize_context<T: Evaluator>(
    context: &Context,
    mut evaluator: T,
) -> Result<MappedContext> {
    context.check_finalized()?;
    evaluator.preprocess(context)?;
    let mut mappings = ContextMappings::default();
    let output_context = create_context()?;
    for graph in context.get_graphs() {
        let (_const_context, const_graph) = graph_in_new_context(graph.clone())?;
        let node_mapping1 =
            optimize_graph_constants(graph.clone(), const_graph.clone(), &mut evaluator)?;
        const_graph.finalize()?;

        let (_meta_context, meta_graph) = graph_in_new_context(graph.clone())?;
        let node_mapping2 =
            optimize_graph_meta_operations(const_graph.clone(), meta_graph.clone())?;
        meta_graph.finalize()?;

        let (_dup_context, dup_graph) = graph_in_new_context(graph.clone())?;
        let node_mapping3 = optimize_graph_duplicates(meta_graph.clone(), dup_graph.clone())?;
        dup_graph.finalize()?;

        let final_graph = add_graph_to_context(output_context.clone(), graph.clone())?;
        let node_mapping4 = optimize_graph_dangling_nodes(dup_graph.clone(), final_graph.clone())?;
        final_graph.finalize()?;

        if graph == context.get_main_graph()? {
            final_graph.set_as_main()?;
        }

        let mapping = ContextMappings::new_from_chain(&[
            node_mapping1,
            node_mapping2,
            node_mapping3,
            node_mapping4,
        ]);
        mappings.extend(mapping);
    }
    output_context.finalize()?;

    Ok(MappedContext::new_with_mappings(
        context.clone(),
        output_context,
        mappings,
    ))
}

fn add_graph_to_context(context: Context, source_graph: Graph) -> Result<Graph> {
    let new_graph = context.create_graph()?;
    for annotation in source_graph.get_annotations()? {
        new_graph.add_annotation(annotation)?;
    }
    Ok(new_graph)
}

fn graph_in_new_context(source_graph: Graph) -> Result<(Context, Graph)> {
    let context = create_context()?;
    let graph = add_graph_to_context(context.clone(), source_graph)?;
    Ok((context, graph))
}

#[doc(hidden)]
pub fn stress_test<T: Evaluator>(
    c1: Context,
    c2: Context,
    ip_evaluator1: T,
    ip_evaluator2: T,
) -> Result<()> {
    let seed = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F";
    let mut prng = PRNG::new(Some(*seed)).unwrap();
    let mut evaluator1 = ip_evaluator1;
    evaluator1.preprocess(&c1).unwrap();
    let mut evaluator2 = ip_evaluator2;
    evaluator2.preprocess(&c2).unwrap();
    for _ in 0..10 {
        let mut inputs = vec![];
        for node in c1.get_main_graph().unwrap().get_nodes() {
            if let Operation::Input(t) = node.get_operation() {
                inputs.push(prng.get_random_value(t).unwrap());
            }
        }
        let result1 = evaluator1
            .evaluate_graph(c1.get_main_graph().unwrap(), inputs.clone())
            .unwrap();
        let result2 = evaluator2
            .evaluate_graph(c2.get_main_graph().unwrap(), inputs.clone())
            .unwrap();
        assert_eq!(result1, result2);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{scalar_type, UINT64};
    use crate::data_values::Value;
    use crate::evaluators::simple_evaluator::SimpleEvaluator;
    use crate::graphs::util::simple_context;

    #[test]
    fn test_simple() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n = i1.add(i2)?;
                n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)
            })?;

            let optimized_c = optimize_context(&c, SimpleEvaluator::new(None)?)?.get_context();
            stress_test(
                c,
                optimized_c,
                SimpleEvaluator::new(None).unwrap(),
                SimpleEvaluator::new(None).unwrap(),
            )?;
            Ok(())
        }()
        .unwrap();
    }
}
