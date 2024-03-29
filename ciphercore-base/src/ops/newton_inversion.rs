//! Multiplicative inversion via [the Newton-Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method#Multiplicative_inverses_of_numbers_and_power_series).
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{Type, INT64, UINT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph};

use serde::{Deserialize, Serialize};

use super::utils::{constant_scalar, inverse_initial_approximation, multiply_fixed_point};

/// A structure that defines the custom operation NewtonInversion that computes an inversion of a number via [the Newton-Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method#Multiplicative_inverses_of_numbers_and_power_series).
///
/// In particular, this operation computes an approximation of 2<sup>denominator_cap_2k</sup> / input.
///
/// Input must be of the scalar type UINT64 or INT64 and be in (0, 2<sup>denominator_cap_2k - 1</sup>) range.
/// The input is also assumed to be small enough (less than 2<sup>32</sup>), otherwise integer overflows
/// are possible, yielding incorrect results.
///
/// Optionally, an initial approximation for the Newton-Raphson method can be provided.
/// In this case, the operation might be faster and of lower depth, however, it must be guaranteed that
/// 2<sup>denominator_cap_2k - 1</sup> <= input * initial_approximation < 2<sup>denominator_cap_2k + 1</sup>.
///
/// # Custom operation arguments
///
/// - Node containing an unsigned or signed 64-bit array or scalar to invert
/// - Negative values are currently unsupported as sign extraction is quite expensive
/// - (optional) Node containing an array or scalar that serves as an initial approximation of the Newton-Raphson method
///
/// # Custom operation returns
///
/// New NewtonInversion node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, UINT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::newton_inversion::NewtonInversion;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], UINT64);
/// let n1 = g.input(t.clone()).unwrap();
/// let guess_n = g.input(t.clone()).unwrap();
/// let n2 = g.custom_op(CustomOperation::new(NewtonInversion {iterations: 10, denominator_cap_2k: 4}), vec![n1, guess_n]).unwrap();
///
// TODO: generalize to other types.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct NewtonInversion {
    /// Number of iterations of the Newton-Raphson method; rule of thumb is to set it to 1 + log(`denominator_cap_2k`)
    pub iterations: u64,
    /// Number of output bits that are approximated
    pub denominator_cap_2k: u64,
}

#[typetag::serde]
impl CustomOperationBody for NewtonInversion {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 && arguments_types.len() != 2 {
            return Err(runtime_error!(
                "Invalid number of arguments for NewtonDivision"
            ));
        }
        let t = arguments_types[0].clone();
        if !t.is_scalar() && !t.is_array() {
            return Err(runtime_error!(
                "Divisor in NewtonDivision must be a scalar or an array"
            ));
        }
        let sc = t.get_scalar_type();
        if sc != UINT64 && sc != INT64 {
            return Err(runtime_error!(
                "Divisor in NewtonDivision must consist of either INT64s or UINT64s"
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

        let g_initial_approximation =
            inverse_initial_approximation(&context, t.clone(), self.denominator_cap_2k)?;
        let g = context.create_graph()?;
        let divisor = g.input(t.clone())?;
        let mut approximation = if has_initial_approximation {
            g.input(t)?
        } else if self.denominator_cap_2k == 0 {
            g.ones(t)?
        } else {
            g.call(g_initial_approximation, vec![divisor.clone()])?
        };
        // Now, we do Newton approximation for computing 1 / x, where x = divisor / (2 ** cap).
        // The formula for the Newton method is x_{i + 1} = x_i * (2 - d * x_i).
        let two_power_cap_plus_one = constant_scalar(&g, 1 << (self.denominator_cap_2k + 1), sc)?;
        for _ in 0..self.iterations {
            let x = approximation;
            let mult = two_power_cap_plus_one.subtract(x.multiply(divisor.clone())?)?;
            approximation = multiply_fixed_point(mult, x, self.denominator_cap_2k)?;
        }
        approximation.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "NewtonDivision(iterations={}, cap=2**{})",
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
        divisor: u64,
        initial_approximation: Option<u64>,
        st: ScalarType,
    ) -> Result<Value> {
        let c = simple_context(|g| {
            let i = g.input(scalar_type(st))?;
            if let Some(approx) = initial_approximation {
                let approx_const = constant_scalar(g, approx, st)?;
                g.custom_op(
                    CustomOperation::new(NewtonInversion {
                        iterations: 5,
                        denominator_cap_2k: 10,
                    }),
                    vec![i, approx_const],
                )
            } else {
                g.custom_op(
                    CustomOperation::new(NewtonInversion {
                        iterations: 5,
                        denominator_cap_2k: 10,
                    }),
                    vec![i],
                )
            }
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_scalar(divisor, st)?],
        )?;
        Ok(result)
    }

    fn array_division_helper(divisor: Vec<u64>, st: ScalarType) -> Result<Vec<u64>> {
        let array_t = array_type(vec![divisor.len() as u64], st);
        let c = simple_context(|g| {
            let i = g.input(array_t.clone())?;
            g.custom_op(
                CustomOperation::new(NewtonInversion {
                    iterations: 5,
                    denominator_cap_2k: 10,
                }),
                vec![i],
            )
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_flattened_array(&divisor, st)?],
        )?;
        result.to_flattened_array_u64(array_t)
    }

    #[test]
    fn test_newton_division_scalar() {
        let div_v = vec![1, 2, 3, 123, 300, 500, 700];
        for i in div_v {
            assert!(
                (scalar_division_helper(i, None, UINT64)
                    .unwrap()
                    .to_u64(UINT64)
                    .unwrap() as i64
                    - 1024 / i as i64)
                    .abs()
                    <= 1
            );

            assert!(
                (scalar_division_helper(i, None, INT64)
                    .unwrap()
                    .to_i64(INT64)
                    .unwrap()
                    - 1024 / i as i64)
                    .abs()
                    <= 1
            );
        }
    }

    #[test]
    fn test_newton_division_array() {
        let arr = vec![23, 32, 57, 71, 183, 555];
        let div = array_division_helper(arr.clone(), UINT64).unwrap();
        let i_div = array_division_helper(arr.clone(), INT64).unwrap();
        for i in 0..arr.len() {
            assert!((div[i] as i64 - 1024 / arr[i] as i64).abs() <= 1);
            assert!((i_div[i] as i64 - 1024 / arr[i] as i64).abs() <= 1);
        }
    }

    #[test]
    fn test_newton_division_with_initial_guess() {
        for i in [1, 2, 3, 123, 300, 500, 700] {
            let mut initial_guess = 1;
            while initial_guess * i * 2 < 1024 {
                initial_guess *= 2;
            }
            assert!(
                (scalar_division_helper(i, Some(initial_guess), UINT64)
                    .unwrap()
                    .to_u64(UINT64)
                    .unwrap() as i64
                    - 1024 / i as i64)
                    .abs()
                    <= 1
            );
            assert!(
                (scalar_division_helper(i, Some(initial_guess), INT64)
                    .unwrap()
                    .to_i64(INT64)
                    .unwrap()
                    - 1024 / i as i64)
                    .abs()
                    <= 1
            );
        }
    }

    #[test]
    fn test_newton_inversion_compiles_end2end() -> Result<()> {
        let c = simple_context(|g| {
            let i = g.input(scalar_type(INT64))?;
            g.custom_op(
                CustomOperation::new(NewtonInversion {
                    iterations: 5,
                    denominator_cap_2k: 10,
                }),
                vec![i],
            )
        })?;
        let inline_config = InlineConfig {
            default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            ..Default::default()
        };
        let instantiated_context = run_instantiation_pass(c)?.get_context();
        let inlined_context =
            inline_operations(&instantiated_context, inline_config.clone())?.get_context();
        let _unused = prepare_for_mpc_evaluation(
            &inlined_context,
            vec![vec![IOStatus::Shared]],
            vec![vec![]],
            inline_config,
        )?;
        Ok(())
    }
}
