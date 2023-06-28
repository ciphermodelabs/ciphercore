use crate::custom_ops::ContextMappings;
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
) -> Result<ContextMappings> {
    graph.check_finalized()?;
    let mut constant_cache = HashMap::<ConstantKey, Node>::new();
    let mut mapping = ContextMappings::default();
    let mut constant_nodes = HashMap::<Node, Value>::new();
    let graph_output_node = graph.get_output_node()?;
    for node in graph.get_nodes() {
        if !node.get_graph_dependencies().is_empty() {
            return Err(runtime_error!(
                "Constant optimization works only on fully inlined graphs."
            ));
        }
        let mut resolve_const = |t: Type, val: Value, name: Option<String>| -> Result<Node> {
            let key = ConstantKey {
                t: t.clone(),
                v: val.clone(),
            };
            if let std::collections::hash_map::Entry::Vacant(e) = constant_cache.entry(key.clone())
            {
                let constant_node = out_graph.constant(t, val)?;
                if let Some(name) = name {
                    constant_node.set_name(&name)?;
                }
                e.insert(constant_node.clone());
                Ok(constant_node)
            } else {
                Ok(constant_cache.get(&key).unwrap().clone())
            }
        };
        let op = node.get_operation();
        let new_node = match op {
            Operation::Constant(t, val) => {
                if !node.get_annotations()?.is_empty() {
                    return Err(runtime_error!(
                        "Constant optimization with annotations on const nodes in not supported"
                    ));
                }
                let value_ptr = evaluator.evaluate_node(node.clone(), vec![])?;
                constant_nodes.insert(node.clone(), value_ptr);
                resolve_const(t, val, node.get_name()?)?
            }
            op => {
                let mut deps = vec![];
                let mut is_const_node = op.is_const_optimizable()?;
                for dep in node.get_node_dependencies() {
                    deps.push(mapping.get_node(&dep));
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
                    resolve_const(node.get_type()?, value_ptr.clone(), node.get_name()?)?
                } else {
                    let result = out_graph.add_node_with_type(
                        deps,
                        vec![],
                        node.get_operation(),
                        node.get_type()?,
                    )?;
                    for annotation in node.get_annotations()? {
                        result.add_annotation(annotation)?;
                    }
                    copy_node_name(node.clone(), result.clone())?;
                    result
                }
            }
        };

        if node == graph_output_node {
            new_node.set_as_output()?;
        }
        mapping.insert_node(node, new_node);
    }
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{array_type, scalar_type, UINT64};
    use crate::evaluators::simple_evaluator::SimpleEvaluator;
    use crate::graphs::create_context;
    use crate::graphs::util::simple_context;
    use crate::graphs::{contexts_deep_equal, Context};

    fn optimize_context(c: &Context) -> Result<Context> {
        let mut evaluator = SimpleEvaluator::new(None)?;
        evaluator.preprocess(c)?;
        let new_c = create_context()?;
        let new_g = new_c.create_graph()?;
        optimize_graph_constants(c.get_main_graph()?, new_g.clone(), &mut evaluator)?;
        new_g.finalize()?;
        new_g.set_as_main()?;
        new_c.finalize()?;
        Ok(new_c)
    }

    #[test]
    fn test_no_duplicates() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.input(scalar_type(UINT64))?;
                let i2 = g.input(scalar_type(UINT64))?;
                let n = i1.add(i2)?;
                n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)
            })?;
            assert!(contexts_deep_equal(&optimize_context(&c)?, &c));
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
                let r1 = g.random(scalar_type(UINT64))?;
                let r2 = g.random_permutation(5)?;
                let r3 = r2.cuckoo_to_permutation()?;
                let r4 = r2.decompose_switching_map(5)?;
                let o1 = n.add(g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?)?;
                g.create_tuple(vec![o1.add(r1)?, r3, r4])
            })?;
            assert!(contexts_deep_equal(&optimize_context(&c)?, &c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_zeros() {
        || -> Result<()> {
            let c = simple_context(|g| g.zeros(array_type(vec![1000, 1000], UINT64)))?;
            assert!(contexts_deep_equal(&optimize_context(&c)?, &c));
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_ones() {
        || -> Result<()> {
            let c = simple_context(|g| g.ones(array_type(vec![1000, 1000], UINT64)))?;
            assert!(contexts_deep_equal(&optimize_context(&c)?, &c));
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

            let new_c = optimize_context(&c)?;
            assert!(!contexts_deep_equal(&new_c, &c));
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

            let new_c = optimize_context(&c)?;
            assert!(!contexts_deep_equal(&new_c, &c));
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

            let new_c = optimize_context(&c)?;
            assert!(!contexts_deep_equal(&new_c, &c));
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
