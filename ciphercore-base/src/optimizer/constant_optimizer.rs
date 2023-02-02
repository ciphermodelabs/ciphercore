use crate::data_types::Type;
use crate::data_values::Value;
use crate::errors::Result;
use crate::evaluators::Evaluator;
use crate::graphs::{copy_node_name, Graph, Node, Operation};
use std::cmp::Eq;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
struct ConstantKey {
    t: Type,
    v: Value,
}

impl Hash for ConstantKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.t.hash(state);
        self.v.deep_hash(state);
    }
}

impl PartialEq for ConstantKey {
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t && self.v == other.v
    }
}

impl Eq for ConstantKey {}

/// This optimization assumes that the graph is fully inlined.
/// It applies the following optimizations to constants in the graph:
/// -- if some node can be precomputed, precompute it;
/// -- if some constant node is repeated multiple times, use only the first occurrence of it, and remove the rest.
/// The names of remaining nodes are preserved.
/// If a constant node is precomputed, its name is taken from the output node of its precomputation.
pub(super) fn optimize_graph_constants(
    graph: Graph,
    out_graph: Graph,
    evaluator: &mut dyn Evaluator,
) -> Result<()> {
    graph.check_finalized()?;
    let mut constant_cache = HashMap::<ConstantKey, Node>::new();
    let mut node_mapping = HashMap::<Node, Node>::new();
    let mut constant_nodes = HashMap::<Node, Value>::new();
    for node in graph.get_nodes() {
        if !node.get_graph_dependencies().is_empty() {
            return Err(runtime_error!(
                "Constant optimization works only on fully inlined graphs."
            ));
        }
        let mut resolve_const = |t: Type, val: Value, name: Result<String>| -> Result<Node> {
            let key = ConstantKey {
                t: t.clone(),
                v: val.clone(),
            };
            if let std::collections::hash_map::Entry::Vacant(e) = constant_cache.entry(key.clone())
            {
                let constant_node = out_graph.constant(t, val)?;
                if name.is_ok() {
                    constant_node.set_name(&(name?))?;
                }
                e.insert(constant_node.clone());
                Ok(constant_node)
            } else {
                Ok(constant_cache.get(&key).unwrap().clone())
            }
        };
        let new_node = match node.get_operation() {
            Operation::Constant(t, val) => {
                if !node.get_annotations()?.is_empty() {
                    return Err(runtime_error!(
                        "Constant optimization with annotations on const nodes in not supported"
                    ));
                }
                let value_ptr = evaluator.evaluate_node(node.clone(), vec![])?;
                constant_nodes.insert(node.clone(), value_ptr);
                resolve_const(t, val, node.get_name())?
            }
            _ => {
                let mut deps = vec![];
                let mut is_const_node = !matches!(node.get_operation(), Operation::Input(_))
                    && !matches!(node.get_operation(), Operation::Random(_));
                for dep in node.get_node_dependencies() {
                    let resolved_dep = node_mapping.get(&dep);
                    match resolved_dep {
                        Some(resolved_dep_node) => deps.push(resolved_dep_node.clone()),
                        None => {
                            panic!("Logic error: unprocessed node in dependencies");
                        }
                    };
                    if !constant_nodes.contains_key(&dep) {
                        is_const_node = false;
                    }
                }
                if is_const_node && node.get_annotations()?.is_empty() {
                    let dep_vals: Vec<Value> = node
                        .get_node_dependencies()
                        .iter()
                        .map(|dep| constant_nodes.get(dep).unwrap().clone())
                        .collect();
                    let value_ptr = evaluator.evaluate_node(node.clone(), dep_vals)?;
                    constant_nodes.insert(node.clone(), value_ptr.clone());
                    resolve_const(node.get_type()?, value_ptr.clone(), node.get_name())?
                } else {
                    let result = out_graph.add_node(deps, vec![], node.get_operation())?;
                    for annotation in node.get_annotations()? {
                        result.add_annotation(annotation)?;
                    }
                    copy_node_name(node.clone(), result.clone())?;
                    result
                }
            }
        };

        if node == graph.get_output_node()? {
            new_node.set_as_output()?;
        }
        node_mapping.insert(node, new_node);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{scalar_type, UINT64};
    use crate::evaluators::simple_evaluator::SimpleEvaluator;
    use crate::graphs::contexts_deep_equal;
    use crate::graphs::create_context;
    use crate::graphs::util::simple_context;

    #[test]
    fn test_no_duplicates() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n = i1.add(i2)?;
                n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)
            })?;

            let mut evaluator = SimpleEvaluator::new(None)?;
            evaluator.preprocess(c.clone())?;
            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_constants(c.get_main_graph()?.clone(), new_g.clone(), &mut evaluator)?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;

            assert!(contexts_deep_equal(new_c, c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_random_is_not_removed() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n = i1.add(i2)?;
                let r = g.random(scalar_type(UINT64))?;
                let o1 = n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)?;
                o1.add(r)
            })?;

            let mut evaluator = SimpleEvaluator::new(None)?;
            evaluator.preprocess(c.clone())?;
            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_constants(c.get_main_graph()?.clone(), new_g.clone(), &mut evaluator)?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;

            assert!(contexts_deep_equal(new_c, c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_constants_simple_deduplication() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(scalar_type(UINT64))?;
                let const1 = g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?;
                const1.set_name("First constant 1")?;
                let n1 = i.add(const1)?;
                let const2 = g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?;
                const2.set_name("Second constant 1")?;
                let n2 = n1.add(const2)?;
                let n3 =
                    n2.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)?;
                let n4 =
                    n3.add(g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?)?;
                n4.add(g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?)
            })?;

            let mut evaluator = SimpleEvaluator::new(None)?;
            evaluator.preprocess(c.clone())?;
            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_constants(c.get_main_graph()?.clone(), new_g.clone(), &mut evaluator)?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;

            assert!(!contexts_deep_equal(new_c.clone(), c));
            let new_o = new_c.get_main_graph()?.get_output_node()?;
            let two1 = new_o.get_node_dependencies()[1].clone();
            let new_n4 = new_o.get_node_dependencies()[0].clone();
            let two2 = new_n4.get_node_dependencies()[1].clone();
            let new_n3 = new_n4.get_node_dependencies()[0].clone();
            let one1 = new_n3.get_node_dependencies()[1].clone();
            let new_n2 = new_n3.get_node_dependencies()[0].clone();
            let one2 = new_n2.get_node_dependencies()[1].clone();
            let new_n1 = new_n2.get_node_dependencies()[0].clone();
            let one3 = new_n1.get_node_dependencies()[1].clone();
            assert!(one1 == one2);
            assert!(one1 == one3);
            assert!(one1 != two1);
            assert!(two1 == two2);
            // Check names
            let new_const1 = new_c.retrieve_node(new_c.get_main_graph()?, "First constant 1");
            assert!(new_const1.is_ok());
            assert_eq!(
                new_const1?.get_operation(),
                Operation::Constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)
            );
            let new_const2 = new_c.retrieve_node(new_c.get_main_graph()?, "Second constant 1");
            assert!(new_const2.is_err());
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_constants_simple_arithmetic() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(scalar_type(UINT64))?;
                let const1 = g.constant(scalar_type(UINT64), Value::from_scalar(4, UINT64)?)?;
                const1.set_name("First constant")?;
                let n1 = i.add(const1)?;
                let const2 = g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?;
                let const3 = g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?;
                let const4 = const2.add(const3)?;
                const4.set_name("Fourth constant")?;
                n1.add(const4)
            })?;

            let mut evaluator = SimpleEvaluator::new(None)?;
            evaluator.preprocess(c.clone())?;
            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_constants(c.get_main_graph()?.clone(), new_g.clone(), &mut evaluator)?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;

            assert!(!contexts_deep_equal(new_c.clone(), c));
            let new_o = new_c.get_main_graph()?.get_output_node()?;
            let four1 = new_o.get_node_dependencies()[1].clone();
            let new_n1 = new_o.get_node_dependencies()[0].clone();
            let four2 = new_n1.get_node_dependencies()[1].clone();
            assert!(four1 == four2);

            // Check names
            let new_const1 = new_c.retrieve_node(new_c.get_main_graph()?, "First constant");
            assert!(new_const1.is_ok());
            assert_eq!(
                new_const1?.get_operation(),
                Operation::Constant(scalar_type(UINT64), Value::from_scalar(4, UINT64)?)
            );
            let new_const4 = new_c.retrieve_node(new_c.get_main_graph()?, "Fourth constant");
            assert!(new_const4.is_err());
            Ok(())
        }()
        .unwrap();
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(scalar_type(UINT64))?;
                let const1 = g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?;
                let const2 = g.constant(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?;
                let const3 = const1.add(const2)?;
                const3.set_name("First constant")?;
                let n = i.add(const3)?;
                let const4 = g.constant(scalar_type(UINT64), Value::from_scalar(4, UINT64)?)?;
                const4.set_name("Fourth constant")?;
                n.add(const4)
            })?;

            let mut evaluator = SimpleEvaluator::new(None)?;
            evaluator.preprocess(c.clone())?;
            let new_c = create_context()?;
            let new_g = new_c.create_graph()?;
            optimize_graph_constants(c.get_main_graph()?.clone(), new_g.clone(), &mut evaluator)?;
            new_g.finalize()?;
            new_g.set_as_main()?;
            new_c.finalize()?;

            assert!(!contexts_deep_equal(new_c.clone(), c));
            let new_o = new_c.get_main_graph()?.get_output_node()?;
            let four1 = new_o.get_node_dependencies()[1].clone();
            let new_n = new_o.get_node_dependencies()[0].clone();
            let four2 = new_n.get_node_dependencies()[1].clone();
            assert!(four1 == four2);

            // Check names
            let new_const1 = new_c.retrieve_node(new_c.get_main_graph()?, "First constant");
            assert!(new_const1.is_ok());
            assert_eq!(
                new_const1?.get_operation(),
                Operation::Constant(scalar_type(UINT64), Value::from_scalar(4, UINT64)?)
            );
            let new_const4 = new_c.retrieve_node(new_c.get_main_graph()?, "Fourth constant");
            assert!(new_const4.is_err());
            Ok(())
        }()
        .unwrap();
    }
}
