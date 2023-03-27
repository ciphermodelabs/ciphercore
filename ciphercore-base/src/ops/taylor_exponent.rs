//! Exp(x) approximation relying on Taylor series expansion.
use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{array_type, scalar_type, vector_type, Type, BIT, INT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph, SliceElement};
use crate::ops::utils::{pull_out_bits, put_in_bits};

use serde::{Deserialize, Serialize};

use super::comparisons::GreaterThanEqualTo;
use super::utils::{constant_scalar, multiply_fixed_point, zeros_like};

/// A structure that defines the custom operation TaylorExponent that computes an approximate exp(x / (2 ** fixed_precision)) * (2 ** fixed_precision) using Taylor expansion.
///
/// Note that Taylor expansion correcly approximates exp(x) only for positive x, so we have to do A2B to get the MSB.
/// Since we're doing A2B anyway, we can compute exp(integer_part(x)) and exp(fractional_part(x)) separately, computing the former directly from bits, and using Taylor expansion for the latter, getting better precision.
/// See [the Keller-Sun paper, Algorithm 2](https://eprint.iacr.org/2022/933.pdf) for more details.
///
/// So far this operation supports only INT64 scalar type.
///
/// # Custom operation arguments
///
/// - Node containing a signed 64-bit array or scalar to compute the exponent
///
/// # Custom operation returns
///
/// New TaylorExponent node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, INT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::taylor_exponent::TaylorExponent;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], INT64);
/// let x = g.input(t.clone()).unwrap();
/// let n2 = g.custom_op(CustomOperation::new(TaylorExponent {taylor_terms: 5, fixed_precision_points: 4}), vec![x]).unwrap();
///
// TODO: generalize to other types.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct TaylorExponent {
    /// Number of terms from the Taylor expansion to consider (5 is typically enough).
    pub taylor_terms: u64,
    /// Assume that we're operating in fixed precision arithmetic with denominator 2 ** fixed_precision_points.
    pub fixed_precision_points: u64,
}

#[typetag::serde]
impl CustomOperationBody for TaylorExponent {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!(
                "Invalid number of arguments for TaylorExponent"
            ));
        }
        let t = arguments_types[0].clone();
        if !t.is_scalar() && !t.is_array() {
            return Err(runtime_error!(
                "Argument in TaylorExponent must be a scalar or an array"
            ));
        }
        let sc = t.get_scalar_type();
        if sc != INT64 {
            return Err(runtime_error!(
                "Argument in TaylorExponent must consist of INT64's"
            ));
        }
        if self.fixed_precision_points > 15 {
            return Err(runtime_error!("fixed_precision_points is too large."));
        }

        let bit_type = if t.is_scalar() {
            scalar_type(BIT)
        } else {
            array_type(t.get_shape(), BIT)
        };

        let g = context.create_graph()?;
        let arg = g.input(t.clone())?;
        // Below, we compute 2 ** (arg / ln(2)) rather than exp(arg).
        // `x` is arg * ln(2).
        let one_over_ln2_int = (((1 << self.fixed_precision_points) as f64) / 2.0_f64.ln()) as u64;
        let one_over_ln2 = constant_scalar(&g, one_over_ln2_int, sc.clone())?;
        let x = multiply_fixed_point(arg, one_over_ln2, self.fixed_precision_points)?;

        let binary_x = x.a2b()?;
        let x_bits = pull_out_bits(binary_x.clone())?;
        let msb = x_bits.get(vec![63])?;

        // STAGE 1: compute exp(integer part of the argument).
        // Note that if we're looking at the int part, we're computing the product of 2 ** (2 ** k) if k'th bit is 1.
        // Since we work with 31-bit fixed-point arithmetic, the exponent is limited from above by 31 - fixed_precision_points.
        let max_exp_bits = (31f64 - self.fixed_precision_points as f64).log2().ceil() as u64;
        let one = g.ones(t)?;
        let mut exp_integer = one.clone();
        for i in self.fixed_precision_points..self.fixed_precision_points + max_exp_bits {
            let bit = x_bits.get(vec![i])?;
            let j = i - self.fixed_precision_points;
            let p2 = constant_scalar(&g, 1_u64 << (1_u64 << j), sc.clone())?;
            // `term` is 1 if bit is not set, and 2 ** (2 ** j) otherwise.
            let term = p2
                .subtract(one.clone())?
                .mixed_multiply(bit.clone())?
                .add(one.clone())?;
            // TODO: this can be optimized to be depth-3 rather than depth-5.
            exp_integer = exp_integer.multiply(term)?;
        }

        // STAGE 2: compute exp(fractional part of the argument).
        // Extract fractional part.
        let exp_fractional = if self.fixed_precision_points == 0 {
            one
        } else {
            let bits_after_point = x_bits.get_slice(vec![
                SliceElement::SubArray(Some(0), Some(self.fixed_precision_points as i64), None),
                SliceElement::Ellipsis,
            ])?;
            let mut bits_before_point_shape = x_bits.get_type()?.get_shape();
            bits_before_point_shape[0] = 64 - self.fixed_precision_points;
            let zero_bits_before_point = g.zeros(array_type(bits_before_point_shape, BIT))?;
            let stacked_frac_bits = g.create_tuple(vec![
                bits_after_point.array_to_vector()?,
                zero_bits_before_point.array_to_vector()?,
            ])?;
            let stacked_type = vector_type(64, bit_type);
            let x_frac = put_in_bits(stacked_frac_bits.reshape(stacked_type)?.vector_to_array()?)?
                .b2a(sc.clone())?;

            // Now, we want 2 ** x = exp(x * ln(2)) = \sum_i (ln(2) * x) ** i / i!
            let mut exp_fractional = zeros_like(x_frac.clone())?;
            let mut coef = constant_scalar(&g, 1 << self.fixed_precision_points, sc.clone())?;
            let ln2_int = (2_f64.ln() * ((1 << self.fixed_precision_points) as f64)) as u64;
            let ln2 = constant_scalar(&g, ln2_int, sc.clone())?;
            let y = multiply_fixed_point(x_frac, ln2, self.fixed_precision_points)?;
            for i in 0..self.taylor_terms {
                exp_fractional = exp_fractional.add(coef.clone())?;
                if i < self.taylor_terms - 1 {
                    coef = coef.multiply(y.clone())?;
                    // We need to divide it by i + 1, and by 2 ** fixed_precision_points, so we combine the two.
                    coef = coef.truncate((i + 1) << self.fixed_precision_points)?;
                }
            }
            exp_fractional
        };

        // STAGE 3: combine the answers, and do exp(-x) if x was negative.
        // No truncation here, since exp_integer is a normal int number, not a fixed-precision one.
        let exp = exp_fractional.multiply(exp_integer)?;
        // If x < 0, then it can be represented as x = -2^max_exp_bits + integer_bits + fractional_bits
        // exp is equal to 2^(integer_bits + fractional_bits).
        // Thus, truncation by 2^(2^max_exp_bits) changes the sign of the exponent.
        let one_over_exp = exp.truncate(1u64 << (1u64 << max_exp_bits))?;
        // Our maximal precision is 15, leading to minimum value around 3e-5. Exp(-10) is 4e-5
        // If x is smaller than -10, return 0.
        let upper_bound_for_inversion =
            constant_scalar(&g, (-10) * (1 << self.fixed_precision_points), sc)?.a2b()?;
        let inversion_overflow_bit = g.custom_op(
            CustomOperation::new(GreaterThanEqualTo {
                signed_comparison: true,
            }),
            vec![binary_x, upper_bound_for_inversion],
        )?;
        let mut result = exp.add(one_over_exp.subtract(exp.clone())?.mixed_multiply(msb)?)?;
        result = result.mixed_multiply(inversion_overflow_bit)?;
        result.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "TaylorExponent(taylor_terms={}, fixed_precision_denom=2**{})",
            self.taylor_terms, self.fixed_precision_points
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;

    fn scalar_helper(arg: i64, precision: u64) -> Result<i64> {
        let c = simple_context(|g| {
            let i = g.input(scalar_type(INT64))?;
            g.custom_op(
                CustomOperation::new(TaylorExponent {
                    taylor_terms: 5,
                    fixed_precision_points: precision,
                }),
                vec![i],
            )
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_scalar(arg, INT64)?],
        )?;
        let res = result.to_i64(INT64)?;
        Ok(res)
    }

    fn array_helper(arg: Vec<i64>) -> Result<Vec<i64>> {
        let array_t = array_type(vec![arg.len() as u64], INT64);
        let c = simple_context(|g| {
            let i = g.input(array_t.clone())?;
            g.custom_op(
                CustomOperation::new(TaylorExponent {
                    taylor_terms: 5,
                    fixed_precision_points: 10,
                }),
                vec![i],
            )
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_flattened_array(&arg, INT64)?],
        )?;
        result.to_flattened_array_i64(array_t)
    }

    #[test]
    fn test_exp_scalar() {
        for i in vec![-10000, -1000, -100, -1, 0, 1, 100, 1000, 10000] {
            let expected = (((i as f64) / 1024.0).exp() * 1024.0) as i64;
            let actual = scalar_helper(i, 10).unwrap();
            let relative_error = ((expected - actual).abs() as f64)
                / (1.0 + f64::max(expected as f64, actual as f64));
            assert!(relative_error <= 0.01);
        }
    }

    #[test]
    fn test_exp_array() {
        let arr = vec![23, 32, 57, 1271, 183, 555, -23, -32, -57, -1271, -183, -555];
        let res = array_helper(arr.clone()).unwrap();
        for i in 0..arr.len() {
            let expected = (((arr[i] as f64) / 1024.0).exp() * 1024.0) as i64;
            let actual = res[i];
            let relative_error = ((expected - actual).abs() as f64)
                / (1.0 + f64::max(expected as f64, actual as f64));
            assert!(relative_error <= 0.01);
        }
    }

    #[test]
    fn test_exp_integer() {
        for i in vec![0, 1, 2, 3, 5] {
            // With zero precision, ln(2) = 1, so it'll compute 2**i instead of exp(i).
            let expected = 1 << i;
            let actual = scalar_helper(i, 0).unwrap();
            let absolute_error = (expected - actual).abs();
            assert!(absolute_error == 0);
        }
    }
}
