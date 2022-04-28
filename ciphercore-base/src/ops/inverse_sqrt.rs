use crate::custom_ops::{CustomOperation, CustomOperationBody, Or};
use crate::data_types::{array_type, scalar_type, Type, BIT, UINT64};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, GraphAnnotation};
use crate::ops::utils::{pull_out_bits, put_in_bits};

use serde::{Deserialize, Serialize};

/// A structure that defines the custom operation InverseSqrt that computes an approximate inverse square root using Newton iterations.
///
/// In particular, this operation computes an approximation of 2<sup>denominator_cap_2k</sup> / sqrt(input).
///
/// Input must be of the scalar type UINT64 and be in (0, 2<sup>2 * denominator_cap_2k - 1</sup>) range.
/// The input is also assumed to be small enough (less than 2<sup>21</sup>), otherwise integer overflows
/// are possible, yielding incorrect results.
///
/// Optionally, an initial approximation for the Newton iterations can be provided.
/// In this case, the operation might be faster and of lower depth, however, it must be guaranteed that
/// 2<sup>2 * denominator_cap_2k - 2</sup> <= input * initial_approximation <= 2<sup>2 * denominator_cap_2k</sup>.
///
/// The following formula for the Newton iterations is used:
///   x_{i + 1} = x_i * (3 / 2 - d / 2 * x_i * x_i).
///
/// # Custom operation arguments
///
/// - Node containing an unsigned 64-bit array or scalar to compute the inverse square root
/// - (optional) Node containing an array or scalar that serves as an initial approximation of the Newton iterations
///
/// # Custom operation returns
///
/// New InverseSqrt node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, UINT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::inverse_sqrt::InverseSqrt;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], UINT64);
/// let n1 = g.input(t.clone()).unwrap();
/// let guess_n = g.input(t.clone()).unwrap();
/// let n2 = g.custom_op(CustomOperation::new(InverseSqrt {iterations: 10, denominator_cap_2k: 4}), vec![n1, guess_n]).unwrap();
///
// TODO: generalize to other types.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct InverseSqrt {
    pub iterations: u64,
    pub denominator_cap_2k: u64,
}

#[typetag::serde]
impl CustomOperationBody for InverseSqrt {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 && arguments_types.len() != 2 {
            return Err(runtime_error!(
                "Invalid number of arguments for InverseSqrt"
            ));
        }
        let t = arguments_types[0].clone();
        if !t.is_scalar() && !t.is_array() {
            return Err(runtime_error!(
                "Divisor in InverseSqrt must be a scalar or an array"
            ));
        }
        if t.get_scalar_type() != UINT64 {
            return Err(runtime_error!(
                "Divisor in InverseSqrt must consist of UINT64's"
            ));
        }
        let has_initial_approximation = arguments_types.len() == 2;
        if has_initial_approximation {
            let divisor_t = arguments_types[1].clone();
            if divisor_t != t {
                return Err(runtime_error!(
                    "Divisor and initial approximation must have the same type."
                ));
            }
        }
        if self.denominator_cap_2k > 31 {
            return Err(runtime_error!("denominator_cap_2k is too large."));
        }

        if self.denominator_cap_2k <= 1 {
            return Err(runtime_error!("denominator_cap_2k is too small."));
        }

        let bit_type = if t.is_scalar() {
            scalar_type(BIT)
        } else {
            array_type(t.get_shape(), BIT)
        };
        // Graph for identifying highest one bit.
        let g_highest_one_bit = context.create_graph()?;
        {
            let input_state = g_highest_one_bit.input(bit_type.clone())?;
            let input_bit = g_highest_one_bit.input(bit_type.clone())?;
            let new_state = g_highest_one_bit.custom_op(
                CustomOperation::new(Or {}),
                vec![input_state.clone(), input_bit],
            )?;
            let output = new_state.add(input_state)?;
            let output_tuple = g_highest_one_bit.create_tuple(vec![new_state, output])?;
            output_tuple.set_as_output()?;
        }
        g_highest_one_bit.add_annotation(GraphAnnotation::AssociativeOperation)?;
        g_highest_one_bit.finalize()?;

        let g = context.create_graph()?;
        let divisor = g.input(t.clone())?;
        let zero_bit = g.constant(bit_type.clone(), Value::zero_of_type(bit_type.clone()))?;
        let mut approximation = if has_initial_approximation {
            g.input(t)?
        } else {
            let divisor_bits = pull_out_bits(divisor.a2b()?)?.array_to_vector()?;
            let mut divisor_bits_reversed = vec![];
            for i in 0..self.denominator_cap_2k {
                // We group pairs of consecutive bits together for the purpose of the initial approximation.
                // Namely, consider divisor to have digits (d_0, ..., d_31) in base-4. Then, if d_k is the highest
                // non-zero digit, our approximation will be 2 ** (cap - k).
                // Indeed, 4 ** k <= divisor < 4 ** (k + 1), so 2 ** (-k - 1) < 1 / sqrt(divisor) < 2 ** -k.
                let index1 = g.constant(
                    scalar_type(UINT64),
                    Value::from_scalar(2 * self.denominator_cap_2k - 2 * i - 1, UINT64)?,
                )?;
                let index2 = g.constant(
                    scalar_type(UINT64),
                    Value::from_scalar(2 * self.denominator_cap_2k - 2 * i - 2, UINT64)?,
                )?;
                let bit1 = divisor_bits.vector_get(index1)?;
                let bit2 = divisor_bits.vector_get(index2)?;
                let bit = g.custom_op(CustomOperation::new(Or {}), vec![bit1, bit2])?;
                divisor_bits_reversed.push(bit);
            }
            let highest_one_bit = g
                .iterate(
                    g_highest_one_bit,
                    zero_bit.clone(),
                    g.create_vector(bit_type.clone(), divisor_bits_reversed)?,
                )?
                .tuple_get(1)?;
            let mut first_approximation_bits = vec![];
            for i in 0..64 {
                let bit = if i >= self.denominator_cap_2k {
                    zero_bit.clone()
                } else {
                    let index = g.constant(scalar_type(UINT64), Value::from_scalar(i, UINT64)?)?;
                    highest_one_bit.vector_get(index)?
                };
                first_approximation_bits.push(bit);
            }
            put_in_bits(
                g.create_vector(bit_type, first_approximation_bits)?
                    .vector_to_array()?,
            )?
            .b2a(UINT64)?
        };
        // Now, we do Newton approximation for computing 1 / sqrt(x), where x = divisor / (2 ** cap).
        // We use F(t) = 1 / (t ** 2) - d;
        // The formula for the Newton method is x_{i + 1} = x_i * (3 / 2 - d / 2 * x_i * x_i).
        let three_halves = g.constant(
            scalar_type(UINT64),
            Value::from_scalar(3 << (self.denominator_cap_2k - 1), UINT64)?,
        )?;
        for _ in 0..self.iterations {
            let x = approximation;
            // We have two terms: 3/2 and divisor * x * x / 2. Since x is multiplied by
            // 2 ** denominator_cap_2k, we should normalize the second term before subtracting from the first one.
            let ax2 = divisor.clone().multiply(x.clone())?.multiply(x.clone())?;
            let ax2_norm = g.truncate(ax2, 1 << (self.denominator_cap_2k + 1))?;

            let mult = three_halves.subtract(ax2_norm)?;
            let new_approximation = mult.multiply(x)?;
            approximation = g.truncate(new_approximation, 1 << self.denominator_cap_2k)?;
        }
        approximation.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "InverseSqrt(iterations={}, cap=2**{})",
            self.iterations, self.denominator_cap_2k
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;

    fn scalar_helper(divisor: u64, initial_approximation: Option<u64>) -> Result<u64> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(scalar_type(UINT64))?;
        let o = if let Some(approx) = initial_approximation {
            let approx_const =
                g.constant(scalar_type(UINT64), Value::from_scalar(approx, UINT64)?)?;
            g.custom_op(
                CustomOperation::new(InverseSqrt {
                    iterations: 5,
                    denominator_cap_2k: 10,
                }),
                vec![i, approx_const],
            )?
        } else {
            g.custom_op(
                CustomOperation::new(InverseSqrt {
                    iterations: 5,
                    denominator_cap_2k: 10,
                }),
                vec![i],
            )?
        };
        o.set_as_output()?;
        g.finalize()?;
        g.set_as_main()?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_scalar(divisor, UINT64)?],
        )?;
        result.to_u64(UINT64)
    }

    fn array_helper(divisor: Vec<u64>) -> Result<Vec<u64>> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let array_t = array_type(vec![divisor.len() as u64], UINT64);
        let i = g.input(array_t.clone())?;
        let o = g.custom_op(
            CustomOperation::new(InverseSqrt {
                iterations: 5,
                denominator_cap_2k: 10,
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
            vec![Value::from_flattened_array(&divisor, UINT64)?],
        )?;
        result.to_flattened_array_u64(array_t)
    }

    #[test]
    fn test_inverse_sqrt_scalar() {
        for i in vec![1, 2, 3, 123, 300, 500, 700] {
            let expected = (1024.0 / (i as f64).powf(0.5)) as i64;
            assert!((scalar_helper(i, None).unwrap() as i64 - expected).abs() <= 1);
        }
    }

    #[test]
    fn test_inverse_sqrt_array() {
        let arr = vec![23, 32, 57, 71, 183, 555];
        let div = array_helper(arr.clone()).unwrap();
        for i in 0..arr.len() {
            let expected = (1024.0 / (arr[i] as f64).powf(0.5)) as i64;
            assert!((div[i] as i64 - expected).abs() <= 1);
        }
    }

    #[test]
    fn test_inverse_sqrt_with_initial_guess() {
        for i in vec![1, 2, 3, 123, 300, 500, 700] {
            let mut initial_guess = 1;
            while initial_guess * initial_guess * i * 4 < 1024 * 1024 {
                initial_guess *= 2;
            }
            let expected = (1024.0 / (i as f64).powf(0.5)) as i64;
            assert!((scalar_helper(i, Some(initial_guess)).unwrap() as i64 - expected).abs() <= 1);
        }
    }
}
