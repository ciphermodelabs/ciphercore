//! Regression tests for broadcasting in custom operations.
//! We don't test correctness, just that it doesn't crash.

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::custom_ops::CustomOperationBody;
    use crate::data_types::INT64;
    use crate::errors::Result;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;
    use crate::ops::adder::BinaryAdd;
    use crate::ops::comparisons::{
        Equal, GreaterThan, GreaterThanEqualTo, LessThan, LessThanEqualTo, NotEqual,
    };
    use crate::ops::fixed_precision::fixed_multiply::FixedMultiply;
    use crate::ops::fixed_precision::fixed_precision_config::FixedPrecisionConfig;
    use crate::ops::goldschmidt_division::GoldschmidtDivision;
    use crate::ops::min_max::Max;
    use crate::ops::min_max::Min;
    use crate::typed_value::TypedValue;
    use crate::typed_value_operations::TypedValueArrayOperations;

    #[test]
    fn test_binary_custom_op_broadcast() -> Result<()> {
        // Broadcasting in custom operations that operate on two bitstrings.
        let values = vec![
            TypedValue::from_scalar(1, INT64)?,
            TypedValue::from_ndarray(array![1].into_dyn(), INT64)?,
            TypedValue::from_ndarray(array![1, 2, 3].into_dyn(), INT64)?,
            TypedValue::from_ndarray(array![[1, 2, 3], [4, 5, 6]].into_dyn(), INT64)?,
        ];
        for v1 in values.iter() {
            for v2 in values.iter() {
                binary_helper(BinaryAdd {}, v1.clone(), v2.clone())?;
                binary_helper(Equal {}, v1.clone(), v2.clone())?;
                binary_helper(NotEqual {}, v1.clone(), v2.clone())?;
                macro_rules! comparison_op {
                    ($custom_op:ident) => {
                        binary_helper(
                            $custom_op {
                                signed_comparison: true,
                            },
                            v1.clone(),
                            v2.clone(),
                        )?;
                    };
                }
                comparison_op!(LessThan);
                comparison_op!(LessThanEqualTo);
                comparison_op!(GreaterThan);
                comparison_op!(GreaterThanEqualTo);
                comparison_op!(Min);
                comparison_op!(Max);
            }
        }
        Ok(())
    }

    fn binary_helper(
        operation: impl CustomOperationBody,
        arg1: TypedValue,
        arg2: TypedValue,
    ) -> Result<()> {
        let c = simple_context(|g| {
            let i1 = g.input(arg1.t)?;
            let i2 = g.input(arg2.t)?;
            let b1 = i1.a2b()?;
            let b2 = i2.a2b()?;
            g.custom_op(CustomOperation::new(operation), vec![b1, b2])
        })?;
        let c = run_instantiation_pass(c)?.context;
        random_evaluate(c.get_main_graph()?, vec![arg1.value, arg2.value])?;
        Ok(())
    }

    #[test]
    fn test_arithemtic_custom_op_broadcast() -> Result<()> {
        // Broadcasting in custom operations that operate on two numbers.
        let values = vec![
            TypedValue::from_scalar(1, INT64)?,
            TypedValue::from_ndarray(array![1].into_dyn(), INT64)?,
            TypedValue::from_ndarray(array![1, 2, 3].into_dyn(), INT64)?,
            TypedValue::from_ndarray(array![[1, 2, 3], [4, 5, 6]].into_dyn(), INT64)?,
        ];
        for v1 in values.iter() {
            for v2 in values.iter() {
                arithmetic_helper(
                    GoldschmidtDivision {
                        iterations: 5,
                        denominator_cap_2k: 15,
                    },
                    v1.clone(),
                    v2.clone(),
                )?;
                arithmetic_helper(
                    FixedMultiply {
                        config: FixedPrecisionConfig {
                            fractional_bits: 10,
                            debug: false,
                        },
                    },
                    v1.clone(),
                    v2.clone(),
                )?;
            }
        }
        Ok(())
    }

    fn arithmetic_helper(
        operation: impl CustomOperationBody,
        arg1: TypedValue,
        arg2: TypedValue,
    ) -> Result<()> {
        let c = simple_context(|g| {
            let i1 = g.input(arg1.t)?;
            let i2 = g.input(arg2.t)?;
            g.custom_op(CustomOperation::new(operation), vec![i1, i2])
        })?;
        let c = run_instantiation_pass(c)?.context;
        random_evaluate(c.get_main_graph()?, vec![arg1.value, arg2.value])?;
        Ok(())
    }
}
