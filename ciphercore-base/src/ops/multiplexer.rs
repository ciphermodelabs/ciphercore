use crate::custom_ops::CustomOperationBody;
use crate::data_types::{Type, BIT};
use crate::errors::Result;
use crate::graphs::{Context, Graph};

use serde::{Deserialize, Serialize};

/// A structure that defines the custom operation Mux that takes three inputs a, b, c and returns b if a is 1 or c if a is 0.
///
/// The inputs should be arrays of bitstrings. This operation is applied elementwise.
///
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if a,b,c are of shapes `[2,3]`, `[1,3]` and `[2,1]`, the resulting array has shape `[2,3]`.
///
/// To use this and other custom operations in computation graphs, see [Graph::custom_op].
///
/// # Custom operation arguments
///
/// - Node containing a binary array or scalar
/// - Node containing a binary array or scalar that will be chosen if the first input is 1
/// - Node containing a binary array or scalar that will be chosen if the first input is 0
///
/// # Custom operation returns
///
/// New Mux node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT, INT32};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::multiplexer::Mux;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t1 = array_type(vec![2, 3], BIT);
/// let t2 = array_type(vec![3], BIT);
/// let n1 = g.input(t1.clone()).unwrap();
/// let n2 = g.input(t1.clone()).unwrap();
/// let n3 = g.input(t2.clone()).unwrap();
/// let n4 = g.custom_op(CustomOperation::new(Mux {}), vec![n1, n2, n3]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Mux {}

#[typetag::serde]
impl CustomOperationBody for Mux {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 3 {
            return Err(runtime_error!("Invalid number of arguments for Mux"));
        }
        let t = arguments_types[0].clone();
        if !t.is_scalar() && !t.is_array() {
            return Err(runtime_error!("Flag for Mux must be a scalar or an array"));
        }
        if t.get_scalar_type() != BIT {
            return Err(runtime_error!("Flag for Mux must consist of bits"));
        }
        let g = context.create_graph()?;
        let i_flag = g.input(arguments_types[0].clone())?;
        let i_choice1 = g.input(arguments_types[1].clone())?;
        let i_choice0 = g.input(arguments_types[2].clone())?;
        i_choice0
            .add(i_flag.multiply(i_choice0.add(i_choice1)?)?)?
            .set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "Mux".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::INT32;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;

    #[test]
    fn test_mux_bits() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i_flag = g.input(Type::Scalar(BIT))?;
            let i_choice1 = g.input(Type::Scalar(BIT))?;
            let i_choice0 = g.input(Type::Scalar(BIT))?;
            let o = g.custom_op(
                CustomOperation::new(Mux {}),
                vec![i_flag, i_choice1, i_choice0],
            )?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g.clone())?;
            c.finalize()?;
            let mapped_c = run_instantiation_pass(c)?;
            for flag in vec![0, 1] {
                for x1 in vec![0, 1] {
                    for x0 in vec![0, 1] {
                        let expected_result = if flag != 0 { x1 } else { x0 };
                        let result = random_evaluate(
                            mapped_c.mappings.get_graph(g.clone()),
                            vec![
                                Value::from_scalar(flag, BIT)?,
                                Value::from_scalar(x1, BIT)?,
                                Value::from_scalar(x0, BIT)?,
                            ],
                        )?
                        .to_u8(BIT)?;
                        assert_eq!(result, expected_result);
                    }
                }
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_mux_broadcast() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i_flag = g.input(Type::Array(vec![3, 1], BIT))?;
            let i_choice1 = g.input(Type::Array(vec![1, 5], BIT))?;
            let i_choice0 = g.input(Type::Array(vec![6, 1, 1], BIT))?;
            let o = g.custom_op(
                CustomOperation::new(Mux {}),
                vec![i_flag, i_choice1, i_choice0],
            )?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g.clone())?;
            c.finalize()?;
            let mapped_c = run_instantiation_pass(c)?;
            let a_flag = vec![0, 1, 1];
            let a_1 = vec![0, 1, 0, 0, 1];
            let a_0 = vec![1, 0, 0, 1, 0, 1];
            let v_flag = Value::from_flattened_array(&a_flag, BIT)?;
            let v_1 = Value::from_flattened_array(&a_1, BIT)?;
            let v_0 = Value::from_flattened_array(&a_0, BIT)?;
            let result = random_evaluate(mapped_c.mappings.get_graph(g), vec![v_flag, v_1, v_0])?
                .to_flattened_array_u64(Type::Array(vec![6, 3, 5], BIT))?;
            for i in 0..6 {
                for j in 0..3 {
                    for k in 0..5 {
                        let r = result[i * 15 + j * 5 + k];
                        let u = a_flag[j];
                        let v = a_1[k];
                        let w = a_0[i];
                        let er = if u != 0 { v } else { w };
                        assert_eq!(r, er);
                    }
                }
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
            let i_flag = g.input(Type::Array(vec![3, 1], BIT))?;
            let i_choice1 = g.input(Type::Array(vec![1, 5], INT32))?;
            let i_choice0 = g.input(Type::Array(vec![6, 1, 1], INT32))?;
            assert!(g
                .custom_op(
                    CustomOperation::new(Mux {}),
                    vec![i_flag, i_choice1, i_choice0]
                )
                .is_err());
            Ok(())
        }()
        .unwrap();

        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i_flag = g.input(Type::Array(vec![3, 1], INT32))?;
            let i_choice1 = g.input(Type::Array(vec![1, 5], BIT))?;
            let i_choice0 = g.input(Type::Array(vec![6, 1, 1], BIT))?;
            assert!(g
                .custom_op(
                    CustomOperation::new(Mux {}),
                    vec![i_flag, i_choice1, i_choice0]
                )
                .is_err());
            Ok(())
        }()
        .unwrap();

        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i_flag = g.input(Type::Array(vec![3, 7], BIT))?;
            let i_choice1 = g.input(Type::Array(vec![1, 5], BIT))?;
            let i_choice0 = g.input(Type::Array(vec![6, 1, 1], BIT))?;
            assert!(g
                .custom_op(
                    CustomOperation::new(Mux {}),
                    vec![i_flag, i_choice1, i_choice0]
                )
                .is_err());
            Ok(())
        }()
        .unwrap();

        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i_flag = g.input(Type::Array(vec![3, 7], BIT))?;
            let i_choice1 = g.input(Type::Array(vec![1, 5], BIT))?;
            let _i_choice0 = g.input(Type::Array(vec![6, 1, 1], BIT))?;
            assert!(g
                .custom_op(CustomOperation::new(Mux {}), vec![i_flag, i_choice1])
                .is_err());
            Ok(())
        }()
        .unwrap();

        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i_flag = g.input(Type::Tuple(vec![]))?;
            let i_choice1 = g.input(Type::Array(vec![1, 5], BIT))?;
            let i_choice0 = g.input(Type::Array(vec![6, 1, 1], BIT))?;
            assert!(g
                .custom_op(
                    CustomOperation::new(Mux {}),
                    vec![i_flag, i_choice1, i_choice0]
                )
                .is_err());
            Ok(())
        }()
        .unwrap();

        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i_flag = g.input(Type::Array(vec![3, 1], BIT))?;
            let i_choice1 = g.input(Type::Array(vec![1, 5], BIT))?;
            let i_choice0 = g.input(Type::Array(vec![6, 1, 1], INT32))?;
            assert!(g
                .custom_op(
                    CustomOperation::new(Mux {}),
                    vec![i_flag, i_choice1, i_choice0]
                )
                .is_err());
            Ok(())
        }()
        .unwrap();
    }
}
