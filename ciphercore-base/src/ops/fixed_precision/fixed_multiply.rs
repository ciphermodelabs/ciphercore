//! Multiplication for the fixed-precision arithmetic.
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{Type, BIT, INT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, SliceElement};
use crate::ops::utils::{ones_like, reduce_mul, unsqueeze};
use crate::typed_value::TypedValue;
use crate::typed_value_operations::TypedValueArrayOperations;

use serde::{Deserialize, Serialize};

use super::fixed_precision_config::FixedPrecisionConfig;

/// Multiplication of numbers in fixed precision.
///
/// In particular, given numbers represented as `x / 2^fractional_bits` and `y / 2^fractional_bits`, this operation returns `x * y / 2^fractional_bits`.
/// This operation supports debug mode, which checks for overflow.
///
/// # Custom operation arguments
///
/// - Fixed precision config
///
/// # Custom operation returns
///
/// Node representing the product of the numbers.
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, INT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::fixed_precision::fixed_multiply::FixedMultiply;
/// # use ciphercore_base::ops::fixed_precision::fixed_precision_config::FixedPrecisionConfig;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], INT64);
/// let a = g.input(t.clone()).unwrap();
/// let b = g.input(t.clone()).unwrap();
/// let config = FixedPrecisionConfig {fractional_bits: 10, debug: false};
/// let res = g.custom_op(CustomOperation::new(FixedMultiply {config}), vec![a, b]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct FixedMultiply {
    pub config: FixedPrecisionConfig,
}

#[typetag::serde]
impl CustomOperationBody for FixedMultiply {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 2 {
            return Err(runtime_error!("FixedMultiply takes two arguments"));
        }
        for arg in arguments_types.iter() {
            if !arg.is_array() && !arg.is_scalar() {
                return Err(runtime_error!(
                    "FixedMultiply expects scalar or array, got {:?}",
                    arg
                ));
            }
            if arg.get_scalar_type() != INT64 {
                return Err(runtime_error!("FixedMultiply expects INT64, got {:?}", arg));
            }
        }

        let g = context.create_graph()?;
        let a = g.input(arguments_types[0].clone())?;
        let b = g.input(arguments_types[1].clone())?;
        let mut a_times_b = a.multiply(b.clone())?;
        if self.config.debug {
            a_times_b = g.assert(
                "Integer overflow".into(),
                is_multiplication_safe_from_overflow(a, b)?,
                a_times_b,
            )?
        }
        let a_times_b_shifted = a_times_b.truncate(self.config.denominator())?;
        a_times_b_shifted.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("FixedMultiply({})", self.config.fractional_bits)
    }
}

/// This function checks whether it is safe to multiply INT64 numbers without overflowing or getting close to it.
/// The primary use-case is inside `FixedMultiply` for the debug mode, but it can also be used in isolation if needed.
pub fn is_multiplication_safe_from_overflow(x: Node, y: Node) -> Result<Node> {
    let x_bits = x.a2b()?;
    let y_bits = y.a2b()?;
    // If `x` and `y` were broadcastable before A2B, they will be broadcastable after A2B.
    // First, let's make sure we don't need to deal with negative numbers.
    // The way we do it is the following: if the MSB bit is set, we flip _all_ bits. This is not exactly negation in the two-complement form, but is close enough.
    let msb_x = x_bits.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(Some(-1), None, None),
    ])?;
    let x_bits = x_bits.add(msb_x)?;
    let msb_y = y_bits.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(Some(-1), None, None),
    ])?;
    let y_bits = y_bits.add(msb_y)?;
    // Now, we're checking the following:
    //   for every pair of indices (i, j) such that i + j >= 56, we check that x[..., i] * y[.., j] == 0.
    // Why 56? Because this guarantees that the resulting product is below 2**63:
    // -- let's consider s in 0..55, and i + j = s;
    // -- max possible contribution for such s is (2 ** s) * (s + 1);
    // -- summing over s <= 55, we get \sum (2 ** s) * (s + 1) = 3963167672086036481, which happens to be lower than 2 ** 63 - 1.
    // -- fwiw, for 56, it would be very close to 2 ** 63 - 1, but still lower. For 57, it is already higher.
    //
    // First, compute xy[..., i, j] = x[..., i] * y[..., j].
    let xy_bits = unsqueeze(x_bits, -1)?.multiply(unsqueeze(y_bits, -2)?)?;
    // Now, mask out pairs of bits such that i + j < 56.
    let mut mask_arr = ndarray::Array2::zeros((64, 64));
    for i in 0..64 {
        for j in 0..64 {
            if i + j >= 56 {
                mask_arr[[i, j]] = 1;
            }
        }
    }
    let mask_tv = TypedValue::from_ndarray(mask_arr.into_dyn(), BIT)?;
    let g = x.get_graph();
    let mask = g.constant(mask_tv.t, mask_tv.value)?;
    let xy_bits = xy_bits.multiply(mask)?;
    // Now, all that remains is to check that `xy_bits` is zero.
    // This is surprisingly non-trivial, we need to reduce it across all dimensions.
    let one = ones_like(xy_bits.clone())?;
    let not_xy_bits = xy_bits.add(one)?;
    let mut reduction_result = not_xy_bits;
    while reduction_result.get_type()?.is_array() {
        reduction_result = reduce_mul(reduction_result)?;
    }
    // `reduction_result` is 1 iff all of `xy_bits` are 0.
    Ok(reduction_result)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::typed_value_operations::ToNdarray;
    use crate::typed_value_operations::TypedValueArrayOperations;

    fn multiply_helper(
        a: TypedValue,
        b: TypedValue,
        config: FixedPrecisionConfig,
    ) -> Result<TypedValue> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let node_a = g.input(a.t.clone())?;
        let node_b = g.input(b.t.clone())?;
        let o = g.custom_op(
            CustomOperation::new(FixedMultiply { config }),
            vec![node_a, node_b],
        )?;
        let t = o.get_type()?;
        o.set_as_output()?;
        g.finalize()?;
        g.set_as_main()?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![a.value, b.value],
        )?;
        TypedValue::new(t, result)
    }

    #[test]
    fn test_multiply_scalars() -> Result<()> {
        let int_config = FixedPrecisionConfig {
            fractional_bits: 0,
            debug: false,
        };
        let two_times_two = multiply_helper(
            TypedValue::from_scalar(2, INT64)?,
            TypedValue::from_scalar(2, INT64)?,
            int_config,
        )?
        .to_u64()?;
        assert_eq!(two_times_two, 4);
        let five_times_six = multiply_helper(
            TypedValue::from_scalar(5, INT64)?,
            TypedValue::from_scalar(6, INT64)?,
            int_config,
        )?
        .to_u64()?;
        assert_eq!(five_times_six, 30);

        let fixed_config = FixedPrecisionConfig {
            fractional_bits: 15,
            debug: false,
        };
        let two_times_two = multiply_helper(
            TypedValue::from_scalar(2 << 15, INT64)?,
            TypedValue::from_scalar(2 << 15, INT64)?,
            fixed_config,
        )?
        .to_u64()?;
        assert_eq!(two_times_two, 4 << 15);
        let five_times_six = multiply_helper(
            TypedValue::from_scalar(5 << 15, INT64)?,
            TypedValue::from_scalar(6 << 15, INT64)?,
            fixed_config,
        )?
        .to_u64()?;
        assert_eq!(five_times_six, 30 << 15);
        Ok(())
    }

    #[test]
    fn test_multiply_negative() -> Result<()> {
        let fixed_config = FixedPrecisionConfig {
            fractional_bits: 15,
            debug: false,
        };
        let two_times_minus_three = multiply_helper(
            TypedValue::from_scalar(2 << 15, INT64)?,
            TypedValue::from_scalar(-3 << 15, INT64)?,
            fixed_config,
        )?
        .to_u64()?;
        assert_eq!(two_times_minus_three as i64, -6 << 15);
        Ok(())
    }

    #[test]
    fn test_multiply_arrays() -> Result<()> {
        let fixed_config = FixedPrecisionConfig {
            fractional_bits: 15,
            debug: false,
        };
        let two_times_x = ToNdarray::<i64>::to_ndarray(&multiply_helper(
            TypedValue::from_scalar(2 << 15, INT64)?,
            TypedValue::from_ndarray(array![1 << 15, 2 << 15, 3 << 15].into_dyn(), INT64)?,
            fixed_config,
        )?)?;
        assert_eq!(two_times_x.into_raw_vec(), vec![2 << 15, 4 << 15, 6 << 15]);
        let x_times_two = ToNdarray::<i64>::to_ndarray(&multiply_helper(
            TypedValue::from_ndarray(array![1 << 15, 2 << 15, 3 << 15].into_dyn(), INT64)?,
            TypedValue::from_scalar(2 << 15, INT64)?,
            fixed_config,
        )?)?;
        assert_eq!(x_times_two.into_raw_vec(), vec![2 << 15, 4 << 15, 6 << 15]);
        let x_times_y = ToNdarray::<i64>::to_ndarray(&multiply_helper(
            TypedValue::from_ndarray(array![1 << 15, 2 << 15, 3 << 15].into_dyn(), INT64)?,
            TypedValue::from_ndarray(array![4 << 15, 5 << 15, 6 << 15].into_dyn(), INT64)?,
            fixed_config,
        )?)?;
        assert_eq!(x_times_y.into_raw_vec(), vec![4 << 15, 10 << 15, 18 << 15]);
        Ok(())
    }

    #[test]
    fn test_multiply_broadcast() -> Result<()> {
        let fixed_config = FixedPrecisionConfig {
            fractional_bits: 15,
            debug: false,
        };
        let x_times_y = ToNdarray::<i64>::to_ndarray(&multiply_helper(
            TypedValue::from_ndarray(array![1 << 15, 2 << 15, 3 << 15].into_dyn(), INT64)?,
            TypedValue::from_ndarray(array![2 << 15].into_dyn(), INT64)?,
            fixed_config,
        )?)?;
        assert_eq!(x_times_y.into_raw_vec(), vec![2 << 15, 4 << 15, 6 << 15]);
        Ok(())
    }

    #[test]
    fn test_multiply_debug_mode_success() -> Result<()> {
        let fixed_config = FixedPrecisionConfig {
            fractional_bits: 15,
            debug: true,
        };
        let two_times_two = multiply_helper(
            TypedValue::from_scalar(2 << 15, INT64)?,
            TypedValue::from_scalar(2 << 15, INT64)?,
            fixed_config,
        )?
        .to_u64()?;
        assert_eq!(two_times_two, 4 << 15);
        Ok(())
    }

    #[test]
    fn test_multiply_debug_mode_fail() -> Result<()> {
        let fixed_config = FixedPrecisionConfig {
            fractional_bits: 15,
            debug: true,
        };
        let err = multiply_helper(
            TypedValue::from_scalar(1 << 30, INT64)?,
            TypedValue::from_scalar(1 << 30, INT64)?,
            fixed_config,
        );
        assert!(err.is_err());
        Ok(())
    }

    fn overflow_helper(a: TypedValue, b: TypedValue) -> Result<bool> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let node_a = g.input(a.t.clone())?;
        let node_b = g.input(b.t.clone())?;
        let o = is_multiplication_safe_from_overflow(node_a, node_b)?;
        let t = o.get_type()?;
        o.set_as_output()?;
        g.finalize()?;
        g.set_as_main()?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![a.value, b.value],
        )?;
        Ok(TypedValue::new(t, result)?.to_u64()? > 0)
    }

    #[test]
    fn test_overflow_check_success() -> Result<()> {
        let one = TypedValue::from_scalar(1, INT64)?;
        let two = TypedValue::from_scalar(2, INT64)?;
        let small_number = TypedValue::from_scalar(4243, INT64)?;
        let two_to_twenty_five = TypedValue::from_scalar(1 << 25, INT64)?;
        let medium_number = TypedValue::from_scalar(71479832, INT64)?;
        let two_to_thirty = TypedValue::from_scalar(1 << 30, INT64)?;
        let two_to_fifty = TypedValue::from_scalar(1_i64 << 50, INT64)?;
        let minus_two = TypedValue::from_scalar(-2, INT64)?;
        let minus_one = TypedValue::from_scalar(-1, INT64)?;
        assert!(overflow_helper(one.clone(), two.clone())?);
        assert!(overflow_helper(one.clone(), minus_two.clone())?);
        assert!(overflow_helper(minus_one.clone(), minus_one)?);
        assert!(overflow_helper(small_number.clone(), small_number.clone())?);
        assert!(overflow_helper(small_number, medium_number.clone())?);
        assert!(overflow_helper(
            two_to_twenty_five.clone(),
            two_to_twenty_five
        )?);
        assert!(overflow_helper(medium_number.clone(), medium_number)?);
        assert!(overflow_helper(two, two_to_thirty.clone())?);
        assert!(overflow_helper(minus_two, two_to_thirty)?);
        assert!(overflow_helper(one, two_to_fifty)?);
        Ok(())
    }

    #[test]
    fn test_overflow_check_fail() -> Result<()> {
        let two_to_twenty_five = TypedValue::from_scalar(1 << 25, INT64)?;
        let two_to_thirty = TypedValue::from_scalar(1 << 30, INT64)?;
        let two_to_fifty = TypedValue::from_scalar(1_i64 << 50, INT64)?;
        let large_number = TypedValue::from_scalar(2363897937439121_i64, INT64)?;
        let minus_two_to_thirty = TypedValue::from_scalar(-1 << 30, INT64)?;
        assert!(!overflow_helper(
            two_to_twenty_five.clone(),
            two_to_fifty.clone()
        )?);
        assert!(!overflow_helper(
            two_to_thirty.clone(),
            two_to_thirty.clone()
        )?);
        assert!(!overflow_helper(
            two_to_thirty.clone(),
            large_number.clone()
        )?);
        assert!(!overflow_helper(large_number.clone(), large_number)?);
        assert!(!overflow_helper(
            minus_two_to_thirty,
            two_to_thirty.clone()
        )?);
        assert!(!overflow_helper(two_to_thirty, two_to_fifty)?);
        Ok(())
    }

    #[test]
    fn test_overflow_check_success_arrays() -> Result<()> {
        let x = TypedValue::from_ndarray(array![1 << 15, 2 << 15, 3 << 15].into_dyn(), INT64)?;
        let y = TypedValue::from_scalar(2, INT64)?;
        assert!(overflow_helper(x.clone(), y.clone())?);
        let x = TypedValue::from_ndarray(array![1 << 15, 2 << 15, 3 << 15].into_dyn(), INT64)?;
        let y = TypedValue::from_ndarray(array![10 << 15, 20 << 15, 30 << 15].into_dyn(), INT64)?;
        assert!(overflow_helper(x.clone(), y.clone())?);
        Ok(())
    }

    #[test]
    fn test_overflow_check_fail_arrays() -> Result<()> {
        let x = TypedValue::from_ndarray(array![1 << 25, 1 << 26, 1 << 27].into_dyn(), INT64)?;
        let y = TypedValue::from_scalar(1 << 30, INT64)?;
        assert!(!overflow_helper(x.clone(), y.clone())?);
        let x = TypedValue::from_ndarray(array![1 << 25, 1 << 26, 1 << 27].into_dyn(), INT64)?;
        let y = TypedValue::from_ndarray(array![1 << 28, 1 << 29, 1 << 30].into_dyn(), INT64)?;
        assert!(!overflow_helper(x.clone(), y.clone())?);
        Ok(())
    }
}
