//! Exp(x) approximation relying on Taylor series expansion.
use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{array_type, scalar_type, vector_type, Type, BIT, INT64};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, SliceElement};
use crate::ops::utils::{pull_out_bits, put_in_bits};

use serde::{Deserialize, Serialize};

use super::comparisons::LessThanEqualTo;
use super::newton_inversion::NewtonInversion;

/// A structure that defines the custom operation TaylorExponent that computes an approximate exp(x / (2 ** fixed_precision)) * (2 ** fixed_precision) using Taylor expansion.
///
/// Note that Taylor expansion correcly approximates exp(x) only for positive x, so we have to do A2B to get the MSB.
/// Since we're doing A2B anyway, we can compute exp(integer_part(x)) and exp(fractional_part(x)) separately, computing the former directly from bits, and using Taylor expansion for the latter, getting better precision.
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
        let arg = g.input(t)?;
        // Below, we compute 2 ** (arg / ln(2)) rather than exp(arg).
        // `x` is arg * ln(2).
        let one_over_ln2_int = (((1 << self.fixed_precision_points) as f64) / 2.0_f64.ln()) as u64;
        let one_over_ln2 = g.constant(
            scalar_type(sc.clone()),
            Value::from_scalar(one_over_ln2_int, sc.clone())?,
        )?;
        let x = arg
            .multiply(one_over_ln2)?
            .truncate(1 << self.fixed_precision_points)?;

        let mut x_bits = pull_out_bits(x.a2b()?)?;
        let msb = x_bits.get(vec![63])?;
        // We're taking an absolute value of `x`. If it is negative, we should invert its bits and add one, but we're skipping the latter.
        // TODO: should we just call a binary adder graph to be more precise?
        x_bits = x_bits.add(msb.clone())?;

        // STAGE 1: compute exp(integer part of the argument).
        // Note that if we're looking at the int part, we're computing the product of 2 ** (2 ** k) if k'th bit is 1.
        // Hence, it doesn't make sense to consider k >= 6.
        let max_exp_bits = 6;
        let one = g.constant(scalar_type(sc.clone()), Value::from_scalar(1, sc.clone())?)?;
        let mut exp_integer = one.clone();
        for i in self.fixed_precision_points..self.fixed_precision_points + max_exp_bits {
            let bit = x_bits.get(vec![i])?;
            let j = i - self.fixed_precision_points;
            let p2 = g.constant(
                scalar_type(sc.clone()),
                Value::from_scalar(1_u64 << (1_u64 << j), sc.clone())?,
            )?;
            // `term` is 1 if bit is not set, and 2 ** (2 ** j) otherwise.
            let term = multiply_bit_and_number(bit.clone(), p2.subtract(one.clone())?)?
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
            let bits_before_point_type = array_type(bits_before_point_shape, BIT);
            let zero_bits_before_point = g.constant(
                bits_before_point_type.clone(),
                Value::zero_of_type(bits_before_point_type),
            )?;
            let stacked_frac_bits = g.create_tuple(vec![
                bits_after_point.array_to_vector()?,
                zero_bits_before_point.array_to_vector()?,
            ])?;
            let stacked_type = vector_type(64, bit_type);
            let x_frac = put_in_bits(stacked_frac_bits.reshape(stacked_type)?.vector_to_array()?)?
                .b2a(sc.clone())?;

            // Now, we want 2 ** x = exp(x * ln(2)) = \sum_i (ln(2) * x) ** i / i!
            let mut exp_fractional =
                g.constant(x_frac.get_type()?, Value::zero_of_type(x_frac.get_type()?))?;
            let mut coef = g.constant(
                scalar_type(sc.clone()),
                Value::from_scalar(1 << self.fixed_precision_points, sc.clone())?,
            )?;
            let ln2_int = (2_f64.ln() * ((1 << self.fixed_precision_points) as f64)) as u64;
            let ln2 = g.constant(
                scalar_type(sc.clone()),
                Value::from_scalar(ln2_int, sc.clone())?,
            )?;
            let y = x_frac
                .multiply(ln2)?
                .truncate(1 << self.fixed_precision_points)?;
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
        // We provide a reasonable initial approximation to avoid bit conversions inside NewtonInversion (we already know highest one bit from the integer part).
        // However, we need bits anyway, for the case of negative argument: we need to compare exp to 2 ** (2 * fixed_precision_points), otherwise it
        // can be to big, overflowing Newton inversion.
        // So we don't provide the initial guess, letting NewtonInversion to call a2b (we rely on optimizer to deduplicate it with the a2b we call here).
        let mut one_over_exp = g.custom_op(
            CustomOperation::new(NewtonInversion {
                // 5 is enough for 64-bit numbers.
                iterations: 5,
                denominator_cap_2k: self.fixed_precision_points * 2,
            }),
            vec![exp.clone()],
        )?;
        // If exp is larger than fixed_precision_points * 2, inversion could overflow. In such case, we should return 0.
        let upper_bound_for_inversion = g
            .constant(
                scalar_type(sc.clone()),
                Value::from_scalar(1 << (2 * self.fixed_precision_points), sc)?,
            )?
            .a2b()?;
        let exp_bits = exp.a2b()?;
        let inversion_overflow_bit = g.custom_op(
            CustomOperation::new(LessThanEqualTo {
                signed_comparison: false,
            }),
            vec![exp_bits, upper_bound_for_inversion],
        )?;
        one_over_exp = multiply_bit_and_number(inversion_overflow_bit, one_over_exp)?;
        let result = exp.add(multiply_bit_and_number(
            msb,
            one_over_exp.subtract(exp.clone())?,
        )?)?;

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

fn multiply_bit_and_number(bit: Node, number: Node) -> Result<Node> {
    // TODO: this function can be made much more efficient.
    let g = bit.get_graph();
    let mut bits = vec![bit.clone()];
    let zero = g.constant(bit.get_type()?, Value::zero_of_type(bit.get_type()?))?;
    for _ in 1..64 {
        bits.push(zero.clone());
    }
    let bit_arithmetic =
        put_in_bits(g.create_vector(bit.get_type()?, bits)?.vector_to_array()?)?.b2a(INT64)?;
    bit_arithmetic.multiply(number)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;

    fn scalar_helper(arg: i64, precision: u64) -> Result<i64> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(scalar_type(INT64))?;
        let o = g.custom_op(
            CustomOperation::new(TaylorExponent {
                taylor_terms: 5,
                fixed_precision_points: precision,
            }),
            vec![i],
        )?;
        o.set_as_output()?;
        g.finalize()?;
        g.set_as_main()?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_scalar(arg, INT64)?],
        )?;
        let res = result.to_i64(INT64)?;
        Ok(res)
    }

    fn array_helper(arg: Vec<i64>) -> Result<Vec<i64>> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let array_t = array_type(vec![arg.len() as u64], INT64);
        let i = g.input(array_t.clone())?;
        let o = g.custom_op(
            CustomOperation::new(TaylorExponent {
                taylor_terms: 5,
                fixed_precision_points: 10,
            }),
            vec![i],
        )?;
        o.set_as_output()?;
        g.finalize()?;
        g.set_as_main()?;
        c.finalize()?;
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
