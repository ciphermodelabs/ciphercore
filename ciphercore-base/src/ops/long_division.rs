//! Long division for bitstrings of arbitrary length.
use crate::broadcast::broadcast_shapes;
use crate::custom_ops::{CustomOperation, CustomOperationBody, Not};
use crate::data_types::{array_type, scalar_type, tuple_type, ArrayShape, Type, BIT};
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, SliceElement};
use crate::ops::multiplexer::Mux;
use crate::ops::utils::unsqueeze;

use serde::{Deserialize, Serialize};

use super::adder::{BinaryAdd, BinaryAddTransposed};
use super::comparisons::Equal;
use super::utils::{prepend_dims, pull_out_bits_pair, put_in_bits};

/// A structure that defines the custom operation LongDivision that computes the quotient and
/// remainder of `dividend` / `divisor`, such that:
///   quotient * divisor + remainder == dividend
///
/// # Custom operation arguments
///
/// - Node containing the dividend as a two's complement length-n bitstring.
/// - Node containing the divisor as a two's complement length-n bitstring.
///
/// Only `n` which are powers of two are supported.
///
/// # Custom operation returns
///
/// Node containing the (quotient, remainder) tuple. Both quotient and remainder are bitstrings with
/// lengths equal to dividend and divisor, respectively.
///
/// # Example
/// ```
/// # use ciphercore_base::graphs::util::simple_context;
/// # use ciphercore_base::data_types::{array_type, INT32};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::long_division::LongDivision;
/// let t = array_type(vec![10, 25], INT32);
/// let c = simple_context(|g| {
///     let input_dividends = g.input(t.clone())?;
///     let input_divisors = g.input(t.clone())?;
///     let binary_dividends = input_dividends.a2b()?;
///     let binary_divisors = input_divisors.a2b()?;
///     let result = g.custom_op(
///         CustomOperation::new(LongDivision {
///             signed: true,
///         }),
///         vec![binary_dividends, binary_divisors],
///     )?;
///     let quotient = result.tuple_get(0)?.b2a(INT32)?;
///     let remainder = result.tuple_get(1)?.b2a(INT32)?;
///     g.create_tuple(vec![quotient, remainder])
/// }).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct LongDivision {
    pub signed: bool,
}

#[typetag::serde]
impl CustomOperationBody for LongDivision {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 2 {
            return Err(runtime_error!(
                "Invalid number of arguments for LongDivision, given {}, expected 2",
                arguments_types.len()
            ));
        }

        let dividend_type = arguments_types[0].clone();
        let divisor_type = arguments_types[1].clone();
        if dividend_type.get_scalar_type() != BIT {
            return Err(runtime_error!(
                "Invalid scalar types for LongDivision: dividend scalar type {}, expected BIT",
                dividend_type.get_scalar_type()
            ));
        }
        if divisor_type.get_scalar_type() != BIT {
            return Err(runtime_error!(
                "Invalid scalar types for LongDivision: divisor scalar type {}, expected BIT",
                dividend_type.get_scalar_type()
            ));
        }
        if !divisor_type.is_array() {
            return Err(runtime_error!("Divisor in LongDivision must be an array"));
        }
        if !dividend_type.is_array() {
            return Err(runtime_error!("Dividend in LongDivision must be an array"));
        }
        let types = Types::new(dividend_type, divisor_type)?;
        let g_iterate = single_iteration_graph(&context, types.clone())?;
        let g = context.create_graph()?;
        let dividend = g.input(types.divident_type.clone())?;
        let divisor = g.input(types.divisor_type.clone())?;

        // We compute abs(dividend) / abs(divisor) first, and adjust the results at the end.
        let (dividend_is_negative, abs_dividend) = abs(dividend, self.signed)?;
        let (divisor_is_negative, abs_divisor) = abs(divisor, self.signed)?;
        let negative_abs_divisor = negative(abs_divisor.clone())?;
        // Pull out dividend bits as first dimesion, and reverse them as we want to process bits
        // starting with the most significant bit. We also pull out divisor bits as it's more
        // efficient to work with them in this form.
        let (dividend_pulled_bits, negative_abs_divisor_pulled_bits) =
            pull_out_bits_pair(abs_dividend, negative_abs_divisor)?;

        let dividend_pulled_bits =
            dividend_pulled_bits.get_slice(vec![SliceElement::SubArray(None, None, Some(-1))])?;

        // Iterate single bit computation over all dividend bits.
        let state = g.create_tuple(vec![
            g.zeros(types.remainder_pulled_bits_type.clone())?,
            broadcast(
                negative_abs_divisor_pulled_bits,
                types.remainder_pulled_bits_type,
            )?,
        ])?;
        let result = g.iterate(g_iterate, state, dividend_pulled_bits.array_to_vector()?)?;
        let remainder = put_in_bits(result.tuple_get(0)?.tuple_get(0)?)?;
        let quotient_pulled_bits = result.tuple_get(1)?.vector_to_array()?;

        // Reverse quotient bits, and put bits back into the last dimension.
        let quotient_pulled_bits =
            quotient_pulled_bits.get_slice(vec![SliceElement::SubArray(None, None, Some(-1))])?;
        let quotient = put_in_bits(quotient_pulled_bits)?;

        let (quotient, remainder) = if self.signed {
            adjust_negative(
                quotient,
                remainder,
                abs_divisor,
                dividend_is_negative,
                divisor_is_negative,
            )?
        } else {
            (quotient, remainder)
        };
        let output = g.create_tuple(vec![quotient, remainder])?;
        output.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("LongDivision(signed={})", self.signed)
    }
}

#[derive(Debug, Clone)]
struct Types {
    divident_type: Type,
    divisor_type: Type,
    remainder_pulled_bits_type: Type,
    quotient_pulled_bit_type: Type,
    dividend_no_bits_type: Type,
    quotient_no_bits_type: Type,
}

impl Types {
    fn new(divident_type: Type, divisor_type: Type) -> Result<Self> {
        let (dividend_no_bits_shape, _dividend_bits) = pop_last_dim(divident_type.get_dimensions());
        let (divisor_no_bits_shape, divisor_bits) = pop_last_dim(divisor_type.get_dimensions());
        let output_no_bits_shape =
            broadcast_shapes(dividend_no_bits_shape.clone(), divisor_no_bits_shape)?;
        let dividend_no_bits_shape =
            prepend_dims(dividend_no_bits_shape, output_no_bits_shape.len())?;
        let remainder_pulled_bits_shape =
            [vec![divisor_bits], output_no_bits_shape.clone()].concat();
        let quotient_pulled_bit_shape = [vec![1], output_no_bits_shape.clone()].concat();
        let quotient_no_bits_shape = output_no_bits_shape;
        Ok(Self {
            divident_type,
            divisor_type,
            remainder_pulled_bits_type: array_type(remainder_pulled_bits_shape, BIT),
            quotient_pulled_bit_type: array_type(quotient_pulled_bit_shape, BIT),
            dividend_no_bits_type: array_type(dividend_no_bits_shape, BIT),
            quotient_no_bits_type: array_type(quotient_no_bits_shape, BIT),
        })
    }
}

fn broadcast(node: Node, want_type: Type) -> Result<Node> {
    let g = node.get_graph();
    if node.get_type()? == want_type {
        Ok(node)
    } else {
        g.zeros(want_type)?.add(node)
    }
}

fn single_iteration_graph(context: &Context, types: Types) -> Result<Graph> {
    // In the state we store (remainder, abs(divisor), -abs(divisor)), with bits dimension pulled
    // out to the outermost level.
    // The remainder is updated in each iteration, while the other two are just passed through all
    // iterations.
    let state_type = tuple_type(vec![
        types.remainder_pulled_bits_type.clone(),
        types.remainder_pulled_bits_type.clone(),
    ]);

    let g = context.create_graph()?;
    // Prepare inputs.
    let old_state = g.input(state_type)?;
    let next_dividend_bit = g.input(types.dividend_no_bits_type.clone())?;
    let remainder = old_state.tuple_get(0)?;
    let minus_divisor = old_state.tuple_get(1)?;

    // Get rid of the most-significant bit of the remainder, and append the next_dividend_bit.
    let remainder = remainder.get_slice(vec![SliceElement::SubArray(None, Some(-1), None)])?;
    // Broadcast the next_dividend_bit before concatenation, if needed.
    let next_dividend_bit = broadcast(next_dividend_bit, types.quotient_pulled_bit_type.clone())?;
    let remainder = g.concatenate(vec![next_dividend_bit, remainder], 0)?;

    // Compute the new remainder.
    let remainder_minus_divisor_with_carry = g.custom_op(
        CustomOperation::new(BinaryAddTransposed { overflow_bit: true }),
        vec![remainder.clone(), minus_divisor.clone()],
    )?;
    let next_quotient_bit = remainder_minus_divisor_with_carry.tuple_get(1)?;
    let remainder_minus_divisor = remainder_minus_divisor_with_carry.tuple_get(0)?;
    let new_remainder = g.custom_op(
        CustomOperation::new(Mux {}),
        vec![
            next_quotient_bit.clone(),
            remainder_minus_divisor,
            remainder,
        ],
    )?;

    let new_state = g.create_tuple(vec![new_remainder, minus_divisor])?;
    let output = g.create_tuple(vec![
        new_state,
        next_quotient_bit.reshape(types.quotient_no_bits_type)?,
    ])?;
    output.set_as_output()?;
    g.finalize()?;
    Ok(g)
}

fn adjust_negative(
    quotient: Node,
    remainder: Node,
    abs_divisor: Node,
    dividend_is_negative: Node,
    divisor_is_negative: Node,
) -> Result<(Node, Node)> {
    // We compute the quotient and remainder using the same logic as numpy's // and %.
    let g = quotient.get_graph();
    let result_is_negative = dividend_is_negative.add(divisor_is_negative.clone())?;
    let remainder_bits = pop_last_dim(remainder.get_type()?.get_dimensions()).1;
    let remainder_is_zero = unsqueeze(
        g.custom_op(
            CustomOperation::new(Equal {}),
            vec![
                remainder.clone(),
                g.zeros(array_type(vec![remainder_bits], BIT))?,
            ],
        )?,
        -1,
    )?;
    // quotient = (-quotient if remainder is 0 else -quotient-1)
    //            if result_is_negative else quotient
    let inverted_quotient = invert_bits(quotient.clone())?; // a.k.a (-quotient-1)
    let negative_quotient = add_one(inverted_quotient.clone())?;
    let quotient = g.custom_op(
        CustomOperation::new(Mux {}),
        vec![
            result_is_negative.clone(),
            g.custom_op(
                CustomOperation::new(Mux {}),
                vec![
                    remainder_is_zero.clone(),
                    negative_quotient,
                    inverted_quotient,
                ],
            )?,
            quotient,
        ],
    )?;
    // positive_remainder = 0 if remainder is 0
    //                      else abs(divisor) - remainder if result_is_negative else remainder
    let positive_remainder = g.custom_op(
        CustomOperation::new(Mux {}),
        vec![
            remainder_is_zero,
            remainder.clone(),
            g.custom_op(
                CustomOperation::new(Mux {}),
                vec![
                    result_is_negative,
                    g.custom_op(
                        CustomOperation::new(BinaryAdd {
                            overflow_bit: false,
                        }),
                        vec![abs_divisor, negative(remainder.clone())?],
                    )?,
                    remainder,
                ],
            )?,
        ],
    )?;
    // If the divisor is negative, we need to return negative divisor, to satisfy:
    //   dividend = divisor * quotient + remainder
    let remainder = g.custom_op(
        CustomOperation::new(Mux {}),
        vec![
            divisor_is_negative,
            negative(positive_remainder.clone())?,
            positive_remainder,
        ],
    )?;
    Ok((quotient, remainder))
}

// Returns the array type with last dimension removed.
fn pop_last_dim(shape: ArrayShape) -> (ArrayShape, u64) {
    let last = shape[shape.len() - 1];
    (shape[..shape.len() - 1].to_vec(), last)
}

fn add_one(binary_num: Node) -> Result<Node> {
    let dims = binary_num.get_type()?.get_dimensions();
    let bits = dims[dims.len() - 1];
    let g = binary_num.get_graph();
    let binary_one = g.concatenate(
        vec![
            g.ones(array_type(vec![1], BIT))?,
            g.zeros(array_type(vec![bits - 1], BIT))?,
        ],
        0,
    )?;
    g.custom_op(
        CustomOperation::new(BinaryAdd {
            overflow_bit: false,
        }),
        vec![binary_num, binary_one],
    )
}

fn invert_bits(binary_num: Node) -> Result<Node> {
    let g = binary_num.get_graph();
    g.custom_op(CustomOperation::new(Not {}), vec![binary_num])
}

// Returns -binary_num, i.e. two's complement of a given number.
fn negative(binary_num: Node) -> Result<Node> {
    add_one(invert_bits(binary_num)?)
}

// Returns 1 where signed number is negative, and 0 elsewhere.
fn is_negative(binary_num: Node) -> Result<Node> {
    binary_num.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(Some(-1), None, None),
    ])
}

// Returns (is_negative(binary_num), abs(binary_num)).
fn abs(binary_num: Node, is_signed: bool) -> Result<(Node, Node)> {
    let g = binary_num.get_graph();
    if is_signed {
        let num_is_negative = is_negative(binary_num.clone())?;
        let abs = g.custom_op(
            CustomOperation::new(Mux {}),
            vec![
                num_is_negative.clone(),
                negative(binary_num.clone())?,
                binary_num,
            ],
        )?;
        Ok((num_is_negative, abs))
    } else {
        Ok((g.zeros(scalar_type(BIT))?, binary_num))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::custom_ops::{run_instantiation_pass, CustomOperation};
    use crate::data_types::{array_type, ScalarType, INT32, INT64, INT8, UINT8};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;
    use crate::typed_value::TypedValue;
    use crate::typed_value_operations::TypedValueArrayOperations;

    #[test]
    fn test_long_division_i32_i8() -> Result<()> {
        let (dividends, divisors, want_q, want_r) = unzip::<i32, i8>(vec![
            (55557, 5, 11111, 2),
            (-55557, 5, -11112, 3),
            (55557, -5, -11112, -3),
            (-55557, -5, 11111, -2),
            (2147483647, 64, 33554431, 63),
            (-2147483648, 64, -33554432, 0),
            (2147483647, 1, 2147483647, 0),
            (-2147483648, 1, -2147483648, 0),
            (-2147483648, -1, -2147483648, 0), // quotient should be positive, but overflows.
            (1, 5, 0, 1),
            (-1, 5, -1, 4),
            (0, 1, 0, 0),
            (0, -1, 0, 0),
            (0, 0, 0, 0), // Division by zero happens to evaluate to this value.
        ]);
        let (q, r) = long_division_helper(dividends.clone(), divisors.clone(), INT32, INT8)?;
        assert_eq!(q.value.to_flattened_array_i32(q.t)?, want_q);
        assert_eq!(r.value.to_flattened_array_i8(r.t)?, want_r);
        Ok(())
    }

    #[test]
    fn test_long_division_u8_u8() -> Result<()> {
        let (dividends, divisors, want_q, want_r) = unzip::<u8, u8>(vec![
            (255, 1, 255, 0),
            (51, 2, 25, 1),
            (85, 6, 14, 1),
            (75, 4, 18, 3),
            (161, 5, 32, 1),
            (173, 6, 28, 5),
            (78, 2, 39, 0),
            (235, 43, 5, 20),
            (244, 228, 1, 16),
            (98, 65, 1, 33),
            (35, 6, 5, 5),
            (187, 249, 0, 187),
            (209, 94, 2, 21),
            (196, 179, 1, 17),
            (112, 213, 0, 112),
            (129, 70, 1, 59),
            (223, 125, 1, 98),
            (0, 1, 0, 0),
            (0, 0, 0, 0), // Division by zero happens to evaluate to this value.
        ]);
        let (q, r) = long_division_helper(dividends.clone(), divisors.clone(), UINT8, UINT8)?;
        assert_eq!(q.value.to_flattened_array_u8(q.t)?, want_q);
        assert_eq!(r.value.to_flattened_array_u8(r.t)?, want_r);
        Ok(())
    }

    #[test]
    fn test_long_division_i64_i64() -> Result<()> {
        let (dividends, divisors, want_q, want_r) = unzip::<i64, i64>(vec![
            (9223372036854775807, 1, 9223372036854775807, 0),
            (-9223372036854775808, 1, -9223372036854775808, 0),
            (-9223372036854775808, -1, -9223372036854775808, 0), // quotient should be positive, but overflows.
            (9223372036854775807, 9223372036854775807, 1, 0),
            (-9223372036854775808, -9223372036854775808, 1, 0),
            (-9223372036854775808, -9223372036854775808, 1, 0),
            (3391070024636615284, 243545908, 13923740507, 102919928),
            (3982195138714201679, -589530672, -6754856580, -156820081),
            (-8836348637758589809, 111540404, -79221056415, 77301851),
            (-2780817202823147876, -882478846, 3151143186, -461104520),
        ]);
        let (q, r) = long_division_helper(dividends.clone(), divisors.clone(), INT64, INT64)?;
        assert_eq!(q.value.to_flattened_array_i64(q.t)?, want_q);
        assert_eq!(r.value.to_flattened_array_i64(r.t)?, want_r);
        Ok(())
    }

    #[test]
    fn test_broadcast_divisor() -> Result<()> {
        let x = TypedValue::from_ndarray(array![[7, 8, 9], [-7, -8, -9]].into_dyn(), INT8)?;
        let y = TypedValue::from_ndarray(array![3].into_dyn(), INT8)?;
        let c = simple_context(|g| {
            let x = g.input(x.t.clone())?.a2b()?;
            let y = g.input(y.t.clone())?.a2b()?;
            let z = g.custom_op(
                CustomOperation::new(LongDivision { signed: true }),
                vec![x, y],
            )?;
            let q = z.tuple_get(0)?.b2a(INT8)?;
            let r = z.tuple_get(1)?.b2a(INT8)?;
            g.create_tuple(vec![q, r])
        })?;
        let c = run_instantiation_pass(c)?.context;
        let g = c.get_main_graph()?;
        let z = random_evaluate(g, vec![x.value, y.value])?.to_vector()?;
        let r_t = array_type(vec![2, 3], INT8);
        let q_t = array_type(vec![2, 3], INT8);
        assert_eq!(z[0].to_flattened_array_i8(r_t)?, [2, 2, 3, -3, -3, -3]);
        assert_eq!(z[1].to_flattened_array_i8(q_t)?, [1, 2, 0, 2, 1, 0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_dividend() -> Result<()> {
        let x = TypedValue::from_ndarray(array![10].into_dyn(), INT8)?;
        let y = TypedValue::from_ndarray(array![[1, 2, 3], [-1, -2, -3]].into_dyn(), INT8)?;
        let c = simple_context(|g| {
            let x = g.input(x.t.clone())?.a2b()?;
            let y = g.input(y.t.clone())?.a2b()?;
            let z = g.custom_op(
                CustomOperation::new(LongDivision { signed: true }),
                vec![x, y],
            )?;
            let q = z.tuple_get(0)?.b2a(INT8)?;
            let r = z.tuple_get(1)?.b2a(INT8)?;
            g.create_tuple(vec![q, r])
        })?;
        let c = run_instantiation_pass(c)?.context;
        let g = c.get_main_graph()?;
        let z = random_evaluate(g, vec![x.value, y.value])?.to_vector()?;
        let r_t = array_type(vec![2, 3], INT8);
        let q_t = array_type(vec![2, 3], INT8);
        assert_eq!(z[0].to_flattened_array_i8(r_t)?, [10, 5, 3, -10, -5, -4]);
        assert_eq!(z[1].to_flattened_array_i8(q_t)?, [0, 0, 1, 0, 0, -2]);
        Ok(())
    }

    fn unzip<A, B>(rows: Vec<(i64, i64, A, B)>) -> (Vec<i64>, Vec<i64>, Vec<A>, Vec<B>) {
        let mut dividends = vec![];
        let mut divisors = vec![];
        let mut quotients = vec![];
        let mut remainders = vec![];
        for (dividend, divisor, quotient, remainder) in rows {
            dividends.push(dividend);
            divisors.push(divisor);
            quotients.push(quotient);
            remainders.push(remainder);
        }
        (dividends, divisors, quotients, remainders)
    }

    fn long_division_helper(
        dividends: Vec<i64>,
        divisors: Vec<i64>,
        dividend_st: ScalarType,
        divisor_st: ScalarType,
    ) -> Result<(TypedValue, TypedValue)> {
        let n = dividends.len();
        if n != divisors.len() {
            return Err(runtime_error!("dividends and divisors length mismatch"));
        }
        if dividend_st.is_signed() != divisor_st.is_signed() {
            return Err(runtime_error!("dividends and divisors signed mismatch"));
        }
        let dividends_t = array_type(vec![n as u64], dividend_st);
        let divisors_t = array_type(vec![n as u64], divisor_st);
        let c = simple_context(|g| {
            let input_dividends = g.input(dividends_t.clone())?;
            let input_divisors = g.input(divisors_t.clone())?;
            let binary_dividends = input_dividends.a2b()?;
            let binary_divisors = input_divisors.a2b()?;
            let result = g.custom_op(
                CustomOperation::new(LongDivision {
                    signed: dividend_st.is_signed(),
                }),
                vec![binary_dividends, binary_divisors],
            )?;
            let quotient = result.tuple_get(0)?.b2a(dividend_st)?;
            let remainder = result.tuple_get(1)?.b2a(divisor_st)?;
            g.create_tuple(vec![quotient, remainder])
        })?;
        let c = run_instantiation_pass(c)?.context;
        let g = c.get_main_graph()?;
        let result = random_evaluate(
            g,
            vec![
                Value::from_flattened_array(&dividends, dividend_st)?,
                Value::from_flattened_array(&divisors, divisor_st)?,
            ],
        )?
        .to_vector()?;
        Ok((
            TypedValue {
                value: result[0].clone(),
                t: dividends_t,
                name: None,
            },
            TypedValue {
                value: result[1].clone(),
                t: divisors_t,
                name: None,
            },
        ))
    }
}
