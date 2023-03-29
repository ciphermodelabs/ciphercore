//! Division via [the Goldschmidt method](https://en.wikipedia.org/wiki/Division_algorithm#Goldschmidt_division).
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{Type, INT64, UINT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph};

use serde::{Deserialize, Serialize};

use super::utils::{constant_scalar, inverse_initial_approximation, multiply_fixed_point};

/// A structure that defines the custom operation GoldSchmidtDivision that computes division of two numbers via [the Goldschmidt method](https://en.wikipedia.org/wiki/Division_algorithm#Goldschmidt_division).
///
/// In particular, this operation computes an approximation of 2<sup>denominator_cap_2k</sup> divdend / divisor.
///
/// Inputs must be of the scalar type UINT64 or INT64 and be in (0, 2<sup>denominator_cap_2k - 1</sup>) range.
/// The divisor is also assumed to be small enough (less than 2<sup>32</sup>), otherwise integer overflows
/// are possible, yielding incorrect results.
///
/// Optionally, an initial approximation for the Goldschmidt method can be provided.
/// In this case, the operation might be faster and of lower depth, however, it must be guaranteed that
/// 2<sup>denominator_cap_2k - 1</sup> <= input * initial_approximation < 2<sup>denominator_cap_2k + 1</sup>.
///
/// # Custom operation arguments
///
/// - Node containing an unsigned or signed 64-bit array or scalar as the dividend.
/// - Node containing an unsigned or signed 64-bit array or scalar as the divisor.
/// - Negative values are currently unsupported as sign extraction is quite expensive
/// - (optional) Node containing an array or scalar that serves as an initial approximation of the GoldSchmidt method
///
/// # Custom operation returns
///
/// New GoldschmidtDivision node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, UINT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::goldschmidt_division::GoldschmidtDivision;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], UINT64);
/// let dividend = g.input(t.clone()).unwrap();
/// let divisor = g.input(t.clone()).unwrap();
/// let guess_n = g.input(t.clone()).unwrap();
/// let n2 = g.custom_op(CustomOperation::new(GoldschmidtDivision {iterations: 10, denominator_cap_2k: 4}), vec![dividend,divisor, guess_n]).unwrap();
///
// TODO: generalize to other types.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct GoldschmidtDivision {
    /// Number of iterations of the Goldschmidt method; rule of thumb is to set it to 1 + log(`denominator_cap_2k`)
    pub iterations: u64,
    /// Number of output bits that are approximated
    pub denominator_cap_2k: u64,
}

#[typetag::serde]
impl CustomOperationBody for GoldschmidtDivision {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 2 && arguments_types.len() != 3 {
            return Err(runtime_error!(
                "Invalid number of arguments for GoldschmidtDivision, given {}, expected 2 or 3",
                arguments_types.len()
            ));
        }

        let dividend_type = arguments_types[0].clone();
        let divisor_type = arguments_types[1].clone();
        if dividend_type.get_scalar_type() != divisor_type.get_scalar_type() {
            return Err(runtime_error!(
                "Invalid scalar types for GoldschmidtDivision: dividend scalr type {} and divisor scalar type {} must be the same",
                dividend_type.get_scalar_type(),
                divisor_type.get_scalar_type()
            ));
        }
        if !divisor_type.is_scalar() && !divisor_type.is_array() {
            return Err(runtime_error!(
                "Divisor in GoldschmidtDivision must be a scalar or an array"
            ));
        }
        if !dividend_type.is_scalar() && !dividend_type.is_array() {
            return Err(runtime_error!(
                "Dividend in GoldschmidtDivision must be a scalar or an array"
            ));
        }

        let sc = dividend_type.get_scalar_type();
        if sc != UINT64 && sc != INT64 {
            return Err(runtime_error!(
                "Divisor in GoldshmidtDivision must consist of either INT64s or UINT64s"
            ));
        }
        let has_initial_approximation = arguments_types.len() == 3;
        if has_initial_approximation {
            let initial_approximation_t = arguments_types[2].clone();
            if initial_approximation_t != divisor_type {
                return Err(runtime_error!(
                    "Divisor and initial approximation must have the same type."
                ));
            }
        }

        let g_initial_approximation =
            inverse_initial_approximation(&context, divisor_type.clone(), self.denominator_cap_2k)?;
        let g = context.create_graph()?;
        let dividend = g.input(dividend_type)?;
        let divisor = g.input(divisor_type.clone())?;
        let approximation = if has_initial_approximation {
            g.input(divisor_type)?
        } else if self.denominator_cap_2k == 0 {
            g.ones(divisor_type)?
        } else {
            g.call(g_initial_approximation, vec![divisor.clone()])?
        };
        // Now, we do Goldschmidt approximation for computing 1 / x,
        // The formula for Goldschmidt division iteration is
        // a_i = a_{i-1} * w_i
        // b_i = b_{i-1} * w_i
        // w_i = 2 - b_i
        let two_power_cap_plus_one = constant_scalar(&g, 1 << (self.denominator_cap_2k + 1), sc)?;
        let mut w = approximation;
        let mut a = dividend.multiply(w.clone())?;
        let mut b = divisor.multiply(w.clone())?;
        for _ in 0..self.iterations - 1 {
            w = two_power_cap_plus_one.subtract(b.clone())?;
            a = multiply_fixed_point(a.clone(), w.clone(), self.denominator_cap_2k)?;
            b = multiply_fixed_point(b.clone(), w.clone(), self.denominator_cap_2k)?;
        }
        a.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "GoldshmidtDivision(iterations={}, cap=2**{})",
            self.iterations, self.denominator_cap_2k
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::{array_type, scalar_type, ScalarType};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;
    use crate::inline::inline_common::DepthOptimizationLevel;
    use crate::inline::inline_ops::inline_operations;
    use crate::inline::inline_ops::InlineConfig;
    use crate::inline::inline_ops::InlineMode;
    use crate::mpc::mpc_compiler::prepare_for_mpc_evaluation;
    use crate::mpc::mpc_compiler::IOStatus;

    fn scalar_division_helper(
        dividend: u64,
        divisor: u64,
        initial_approximation: Option<u64>,
        st: ScalarType,
    ) -> Result<Value> {
        let c = simple_context(|g| {
            let dividend_node = g.input(scalar_type(st))?;
            let divisor_node = g.input(scalar_type(st))?;
            if let Some(approx) = initial_approximation {
                let approx_const = constant_scalar(&g, approx, st)?;
                g.custom_op(
                    CustomOperation::new(GoldschmidtDivision {
                        iterations: 5,
                        denominator_cap_2k: 10,
                    }),
                    vec![dividend_node, divisor_node, approx_const],
                )
            } else {
                g.custom_op(
                    CustomOperation::new(GoldschmidtDivision {
                        iterations: 5,
                        denominator_cap_2k: 10,
                    }),
                    vec![dividend_node, divisor_node],
                )
            }
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![
                Value::from_scalar(dividend, st)?,
                Value::from_scalar(divisor, st)?,
            ],
        )?;
        Ok(result)
    }

    fn array_division_helper_array_scalar(
        dividend: Vec<u64>,
        divisor: u64,
        st: ScalarType,
    ) -> Result<Vec<u64>> {
        let array_t = array_type(vec![dividend.len() as u64], st);
        let c = simple_context(|g| {
            let dividend_node = g.input(array_t.clone())?;
            let divisor_node = g.input(scalar_type(st))?;
            g.custom_op(
                CustomOperation::new(GoldschmidtDivision {
                    iterations: 5,
                    denominator_cap_2k: 10,
                }),
                vec![dividend_node, divisor_node],
            )
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![
                Value::from_flattened_array(&dividend, st)?,
                Value::from_scalar(divisor, st)?,
            ],
        )?;
        result.to_flattened_array_u64(array_t)
    }

    fn array_division_helper_scalar_array(
        dividend: u64,
        divisor: Vec<u64>,
        st: ScalarType,
    ) -> Result<Vec<u64>> {
        let array_t = array_type(vec![divisor.len() as u64], st);
        let c = simple_context(|g| {
            let dividend_node = g.input(scalar_type(st))?;
            let divisor_node = g.input(array_t.clone())?;
            g.custom_op(
                CustomOperation::new(GoldschmidtDivision {
                    iterations: 5,
                    denominator_cap_2k: 10,
                }),
                vec![dividend_node, divisor_node],
            )
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![
                Value::from_scalar(dividend, st)?,
                Value::from_flattened_array(&divisor, st)?,
            ],
        )?;
        result.to_flattened_array_u64(array_t)
    }

    fn array_division_helper_array_array(
        dividend: Vec<u64>,
        divisor: Vec<u64>,
        st: ScalarType,
    ) -> Result<Vec<u64>> {
        let array_t = array_type(vec![divisor.len() as u64], st);
        let c = simple_context(|g| {
            let dividend_node = g.input(array_t.clone())?;
            let divisor_node = g.input(array_t.clone())?;
            g.custom_op(
                CustomOperation::new(GoldschmidtDivision {
                    iterations: 5,
                    denominator_cap_2k: 10,
                }),
                vec![dividend_node, divisor_node],
            )
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![
                Value::from_flattened_array(&dividend, st)?,
                Value::from_flattened_array(&divisor, st)?,
            ],
        )?;
        result.to_flattened_array_u64(array_t)
    }

    #[test]
    fn test_goldschmidt_division_scalar() {
        let dividend = 123456;
        let div_v = vec![1, 2, 3, 123, 300, 500, 700];
        for i in div_v.clone() {
            let result_int64 = scalar_division_helper(dividend, i, None, INT64)
                .unwrap()
                .to_i64(INT64)
                .unwrap() as i64;
            let result_uint64 = scalar_division_helper(dividend, i, None, UINT64)
                .unwrap()
                .to_u64(UINT64)
                .unwrap() as i64;
            let actual_result = (dividend * (1 << 10) / i) as i64;

            assert!(((result_int64 - actual_result).abs() * 100) / actual_result <= 1);
            assert!(((result_uint64 - actual_result).abs() * 100) / actual_result <= 1);
        }
    }

    #[test]
    fn test_goldschmidt_division_array() {
        let dividends = vec![2300, 3200, 57, 71000, 183293, 55511];
        let divisor = 122;
        let div = array_division_helper_array_scalar(dividends.clone(), divisor, UINT64).unwrap();
        let i_div = array_division_helper_array_scalar(dividends.clone(), divisor, INT64).unwrap();
        let actual_result = dividends
            .iter()
            .map(|x| (x * (1 << 10) / divisor) as i64)
            .collect::<Vec<i64>>();
        for i in 0..dividends.len() {
            let result_int64 = i_div[i] as i64;
            let result_uint64 = div[i] as i64;
            assert!(((result_int64 - actual_result[i]).abs() * 100) / actual_result[i] <= 1);
            assert!(((result_uint64 - actual_result[i]).abs() * 100) / actual_result[i] <= 1);
        }
        let dividend = 1234567;
        let divisors = vec![23, 32, 57, 710, 183, 555];
        let div = array_division_helper_scalar_array(dividend, divisors.clone(), UINT64).unwrap();
        let i_div = array_division_helper_scalar_array(dividend, divisors.clone(), INT64).unwrap();
        let actual_result = divisors
            .iter()
            .map(|x| (dividend * (1 << 10) / x) as i64)
            .collect::<Vec<i64>>();
        for i in 0..dividends.len() {
            let result_int64 = i_div[i] as i64;
            let result_uint64 = div[i] as i64;
            assert!(((result_int64 - actual_result[i]).abs() * 100) / actual_result[i] <= 1);
            assert!(((result_uint64 - actual_result[i]).abs() * 100) / actual_result[i] <= 1);
        }
        let dividends = vec![2300, 3200, 57, 71000, 183293, 55511];
        let divisors = vec![23, 32, 57, 710, 183, 555];
        let div =
            array_division_helper_array_array(dividends.clone(), divisors.clone(), UINT64).unwrap();
        let i_div =
            array_division_helper_array_array(dividends.clone(), divisors.clone(), INT64).unwrap();
        let actual_result = dividends
            .iter()
            .zip(divisors.iter())
            .map(|(x, y)| (*x * (1 << 10) / *y) as i64)
            .collect::<Vec<i64>>();
        for i in 0..dividends.len() {
            let result_int64 = i_div[i] as i64;
            let result_uint64 = div[i] as i64;
            assert!(((result_int64 - actual_result[i]).abs() * 100) / actual_result[i] <= 1);
            assert!(((result_uint64 - actual_result[i]).abs() * 100) / actual_result[i] <= 1);
        }
    }
    #[test]
    fn test_goldschmidt_division_compiles_end2end() -> Result<()> {
        let c = simple_context(|g| {
            let dividend = g.input(scalar_type(INT64))?;
            let divisor = g.input(scalar_type(INT64))?;
            g.custom_op(
                CustomOperation::new(GoldschmidtDivision {
                    iterations: 5,
                    denominator_cap_2k: 10,
                }),
                vec![dividend, divisor],
            )
        })?;
        let inline_config = InlineConfig {
            default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            ..Default::default()
        };
        let instantiated_context = run_instantiation_pass(c)?.get_context();
        let inlined_context = inline_operations(instantiated_context, inline_config.clone())?;
        let _unused = prepare_for_mpc_evaluation(
            inlined_context,
            vec![vec![IOStatus::Shared, IOStatus::Shared]],
            vec![vec![]],
            inline_config,
        )?;
        Ok(())
    }
}
