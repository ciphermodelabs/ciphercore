//! Minimum and maximum operations.
use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{array_type, Type};
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node};

use super::comparisons::GreaterThan;
use super::multiplexer::Mux;

use serde::{Deserialize, Serialize};

/// A structure that defines the custom operation Min that computes the minimum of length-n bitstring arrays elementwise.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2,3]`.
///
/// To use this and other custom operations in computation graphs, see [Graph::custom_op].
///
/// # Custom operation arguments
///
/// - Node containing a binary array or scalar
/// - Node containing a binary array or scalar
///
/// # Custom operation returns
///
/// New Min node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::min_max::Min;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(Min {}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Min {}

/// If `cmp` is an array, add `1` to the shape by reshaping,
/// otherwise, do nothing.
/// This helper function is necessary for min/max, since
/// we need to pad the shape of the result of the comparison
/// in order to be able to call mux later.
fn normalize_cmp(cmp: Node) -> Result<Node> {
    let cmp_type = cmp.get_type()?;
    let normalized_cmp = if cmp_type.is_array() {
        let mut new_shape = cmp_type.get_shape();
        let st = cmp_type.get_scalar_type();
        new_shape.push(1);
        cmp.reshape(array_type(new_shape, st))?
    } else {
        cmp
    };
    Ok(normalized_cmp)
}

#[typetag::serde]
impl CustomOperationBody for Min {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 2 {
            return Err(runtime_error!("Invalid number of arguments for Min"));
        }
        let g = context.create_graph()?;
        let i1 = g.input(arguments_types[0].clone())?;
        let i2 = g.input(arguments_types[1].clone())?;
        let cmp = g.custom_op(
            CustomOperation::new(GreaterThan {
                signed_comparison: false,
            }),
            vec![i1.clone(), i2.clone()],
        )?;
        let normalized_cmp = normalize_cmp(cmp)?;
        let o = g.custom_op(CustomOperation::new(Mux {}), vec![normalized_cmp, i2, i1])?;
        g.set_output_node(o)?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "Min".to_owned()
    }
}

/// A structure that defines the custom operation Max that computes the maximum of length-n bitstring arrays elementwise.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2,3]`.
///
/// To use this and other custom operations in computation graphs, see [Graph::custom_op].
///
/// # Custom operation arguments
///
/// - Node containing a binary array or scalar
/// - Node containing a binary array or scalar
///
/// # Custom operation returns
///
/// New Max node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::min_max::Max;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(Max {}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Max {}

#[typetag::serde]
impl CustomOperationBody for Max {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 2 {
            return Err(runtime_error!("Invalid number of arguments for Max"));
        }
        let g = context.create_graph()?;
        let i1 = g.input(arguments_types[0].clone())?;
        let i2 = g.input(arguments_types[1].clone())?;
        let cmp = g.custom_op(
            CustomOperation::new(GreaterThan {
                signed_comparison: false,
            }),
            vec![i1.clone(), i2.clone()],
        )?;
        let normalized_cmp = normalize_cmp(cmp)?;
        let o = g.custom_op(CustomOperation::new(Mux {}), vec![normalized_cmp, i1, i2])?;
        g.set_output_node(o)?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "Max".to_owned()
    }
}

#[cfg(test)]
mod tests {

    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::{array_type, scalar_type, BIT, UINT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;

    use super::*;

    use std::cmp::{max, min};

    #[test]
    fn test_well_formed() {
        || -> Result<()> {
            let test_data: Vec<(u64, u64)> = vec![
                (31, 32),
                (76543, 76544),
                (0, 1),
                (0, 0),
                (761523, 761523),
                (18446744073709551615u64, 18446744073708999999u64),
            ];
            let context = || -> Result<Context> {
                let c = create_context()?;
                let g = c.create_graph()?;
                let i1 = g.input(scalar_type(UINT64))?.a2b()?;
                let i2 = g.input(scalar_type(UINT64))?.a2b()?;
                let o = g.create_tuple(vec![
                    g.custom_op(CustomOperation::new(Min {}), vec![i1.clone(), i2.clone()])?,
                    g.custom_op(CustomOperation::new(Max {}), vec![i1.clone(), i2.clone()])?,
                ])?;
                g.set_output_node(o)?;
                g.finalize()?;
                c.set_main_graph(g)?;
                c.finalize()?;
                let mapped_c = run_instantiation_pass(c)?;
                Ok(mapped_c.get_context())
            }()?;
            for (u, v) in test_data {
                let minmax = random_evaluate(
                    context.get_main_graph()?,
                    vec![
                        Value::from_scalar(u, UINT64)?,
                        Value::from_scalar(v, UINT64)?,
                    ],
                )?
                .to_vector()?;
                let computed_min = minmax[0].to_u64(UINT64)?;
                let computed_max = minmax[1].to_u64(UINT64)?;
                assert_eq!(min(u, v), computed_min);
                assert_eq!(max(u, v), computed_max);
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_malformed() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.input(scalar_type(UINT64))?.a2b()?;
            assert!(g
                .custom_op(CustomOperation::new(Min {}), vec![i1.clone()])
                .is_err());
            assert!(g
                .custom_op(CustomOperation::new(Max {}), vec![i1.clone()])
                .is_err());
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_vector() {
        || -> Result<()> {
            let context = || -> Result<Context> {
                let c = create_context()?;
                let g = c.create_graph()?;
                let i1 = g.input(array_type(vec![1, 3, 64], BIT))?;
                let i2 = g.input(array_type(vec![3, 1, 64], BIT))?;
                let o = g.create_tuple(vec![
                    g.custom_op(CustomOperation::new(Min {}), vec![i1.clone(), i2.clone()])?,
                    g.custom_op(CustomOperation::new(Max {}), vec![i1.clone(), i2.clone()])?,
                ])?;
                g.set_output_node(o)?;
                g.finalize()?;
                c.set_main_graph(g)?;
                c.finalize()?;
                let mapped_c = run_instantiation_pass(c)?;
                Ok(mapped_c.get_context())
            }()?;
            let a = vec![0, 30, 100];
            let b = vec![10, 50, 150];
            let v = random_evaluate(
                context.get_main_graph()?,
                vec![
                    Value::from_flattened_array(&a, UINT64)?,
                    Value::from_flattened_array(&b, UINT64)?,
                ],
            )?
            .to_vector()?;
            let min_a_b = v[0].to_flattened_array_u64(array_type(vec![3, 3], UINT64))?;
            let max_a_b = v[1].to_flattened_array_u64(array_type(vec![3, 3], UINT64))?;
            assert_eq!(min_a_b, vec![0, 10, 10, 0, 30, 50, 0, 30, 100]);
            assert_eq!(max_a_b, vec![10, 30, 100, 50, 50, 100, 150, 150, 150]);
            Ok(())
        }()
        .unwrap();
    }
}
