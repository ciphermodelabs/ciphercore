//! Various comparison functions for signed and unsigned integers including greater-than, less-than, greater-than-equal-to, less-than-equal-to, equal, not-equal.
use crate::broadcast::broadcast_shapes;
use crate::custom_ops::{CustomOperation, CustomOperationBody, Not, Or};
use crate::data_types::{array_type, scalar_type, tuple_type, vector_type, ArrayShape, Type, BIT};
use crate::errors::Result;
use crate::graphs::*;
use crate::ops::utils::pull_out_bits;
use crate::ops::utils::validate_arguments_in_broadcast_bit_ops;
use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::utils::zeros;

/// Given an array_shape, this function returns the array's trailing
/// axis' (or the innermost/last dimension's) length
fn get_innermost_dim_len(array_shape: ArrayShape) -> u64 {
    array_shape[array_shape.len() - 1]
}

/// Given shape of graph inputs, this function returns:
/// - bit_vect_len: Bit vector length after input vectorization, which is to be used in cust_op graph later.
/// - permuted_shape: Permutation required for permute_axes graph operation, if argument has single dimension i.e. BIT ScalarType, this would be empty.
/// - new_arg_shape: Shape taken by input once array is vectorized, could be empty if BIT ScalarType is the input.
/// - new_arg_type: Type formed out of input once input is vectorized, could be a ArrayType(_, BIT) or ScalarType(BIT).
/// - mult_dim_op: Value is `true` if argument has two or more dimensions else false
fn preprocess_comparison_args(argument_type: Type) -> Result<(u64, ArrayShape, Type, bool)> {
    let shape = argument_type.get_shape();
    let new_arg_shape: ArrayShape;
    let new_arg_type: Type;
    let mult_dim_op: bool;
    let shape_len: u64 = shape.len() as u64;

    let bit_vect_len = get_innermost_dim_len(shape.clone());
    match shape_len.cmp(&1) {
        Ordering::Equal => {
            new_arg_shape = vec![];
            new_arg_type = scalar_type(BIT);
            mult_dim_op = false;
        }
        Ordering::Greater => {
            new_arg_shape = Vec::from(&shape[0..(shape.len() - 1)]);
            new_arg_type = array_type(new_arg_shape.to_vec(), argument_type.get_scalar_type());
            mult_dim_op = true;
        }
        _ => {
            return Err(runtime_error!(
                "Input argument for custom operation - comparison - is empty"
            ));
        }
    }
    Ok((bit_vect_len, new_arg_shape, new_arg_type, mult_dim_op))
}

/// This trait has to be implemented by any comparison custom operation
#[typetag::serde(tag = "type")]
trait ComparisonCustomOperation: CustomOperationBody {
    /// - This function needs to be implemented if the arguments need to be
    /// validated additionally, i.e. apart from the generic argument validation,
    /// as implemented by `validate_arguments_in_broadcast_bit_ops()`
    /// - Some custom comparison operations may require additional operation-
    /// specific validation
    /// - E.g. see the implementation for this function for `GreaterThan` custom op,
    /// in there, unsigned comparison for single-bit inputs is supported,
    /// whereas signed comparison requires at least two bits (one for MSB). This
    /// signed-operation specific validation is done in the default
    /// `validate_signed_arguments()` function for the trait `NeedsSignedPreprocessing`.
    /// This function's overridden counterpart for `GreaterThan` custom operation
    /// simply makes a call for it.
    fn validate_c_op_specific_arguments(&self, _arguments_types: Vec<Type>) -> Result<()> {
        Ok(())
    }

    /// Returns error if vector comparison custom_ops input is invalid.
    fn validate_comparison_arguments(&self, arguments_types: Vec<Type>) -> Result<()> {
        // Do generic argument validations
        validate_arguments_in_broadcast_bit_ops(arguments_types.clone(), &self.get_name())?;
        // Do custom comparison operation specific argument validation
        self.validate_c_op_specific_arguments(arguments_types)
    }

    /// - This function returns the Node after array_to_vector() Graph operation
    /// for input operation node `ip`.
    /// - While doing so it will permute `ip`, for multidimensional inputs only,
    /// using the permute_axes() Graph operation. Permutation is done to bring
    /// the innermost dimension axis to the outermost dimension as it is assumed
    /// that, prior to permutation, inner dimension corresponds to the bit representation of `ip`.
    /// Inputs -
    /// - ip: Input node to be converted to vector
    fn get_bin_vec(&self, ip: Node) -> Result<Node> {
        // TODO - Could be unified with `pull_out_bits()` or extracted to `utils.rs`
        match ip.get_type()? {
            Type::Array(_, _) => {
                let graph = ip.get_graph();
                Ok(graph.array_to_vector(pull_out_bits(ip)?)?)
            }
            _ => Err(runtime_error!(
                "Invalid Node argument: expected operation Input of ArrayType"
            )),
        }
    }
}

/// This trait would only be implemented by custom comparison operations that
/// support signed comparisons e.g. GreaterThan, LessThan, GreaterThanEqualTo,
/// LessThanEqualTo.
#[typetag::serde(tag = "type")]
trait NeedsSignedPreprocessing: CustomOperationBody + ComparisonCustomOperation {
    fn is_signed_custom_operation(&self) -> bool;

    /// This function validates if the `arguments_types` are suitable for the
    /// intended signed custom operation. E.g. there should be at least `2` bits
    /// i.e. ( magnitude + sign )
    fn validate_signed_arguments(&self, arguments_types: Vec<Type>) -> Result<()> {
        let mut are_valid_inputs: bool = true;
        let mut error_message: String = format!("{}: ", self.get_name());
        match (&arguments_types[0], &arguments_types[1]) {
            (Type::Array(shape0, _), Type::Array(shape1, _)) => {
                let shape0_len = shape0.len();
                let shape1_len = shape1.len();
                if self.is_signed_custom_operation()
                    && shape0_len == 1
                    && shape0[shape0_len - 1] < 2
                {
                    are_valid_inputs = false;
                    error_message.push_str("Signed input0 has less than 2 bits");
                } else if self.is_signed_custom_operation()
                    && shape1_len == 1
                    && shape1[shape1_len - 1] < 2
                {
                    are_valid_inputs = false;
                    error_message.push_str("Signed input1 has less than 2 bits");
                }
            }
            _ => {
                are_valid_inputs = false;
                error_message.push_str("Invalid input argument type, expected Array type");
            }
        }

        if !are_valid_inputs {
            Err(runtime_error!("{}", error_message))
        } else {
            Ok(())
        }
    }

    /// - This function flips the input Array's MSB bit, to enable the signed
    /// comparisons, and returns a vector, whose each component corresponds to
    /// the BIT, i.e. the inner-most, Array dimension
    ///
    /// - Why bit flip is sufficient for obtaining signed comparisons given
    /// unsigned comparison functionality? Please see below example:
    ///
    /// ========================================
    /// |sign bit|  b1|  b0| unsigned|   signed|
    /// |   MSB  |    |    |    value|    value|
    /// |========|====|====|=========|=========|
    /// |       0|   0|   0|        0|        0|
    /// |       0|   0|   1|        1|        1|
    /// |       0|   1|   0|        2|        2|
    /// |       0|   1|   1|        3|        3|
    /// |       1|   0|   0|        4|       -4|
    /// |       1|   0|   1|        5|       -3|
    /// |       1|   1|   0|        6|       -2|
    /// |       1|   1|   1|        7|       -1|
    /// ========================================
    ///
    /// - From the table, it can be seen that simply flipping the Most Significant Bit
    /// followed by doing unsigned comparison operation can provide the result achieved
    /// by performing the signed operation before the flipping.
    ///
    /// - e.g. For both positive inputs,
    /// (2, 3)->(010, 011)-FlipMSB->(110, 111)-unsignedGreaterThan-> false
    /// unsigned comparison over them gives signed result
    ///
    /// - e.g. For positive and negative inputs,
    /// (3, -4)->(011, 100)-FlipMSB->(111, 000)-unsignedGreaterThan-> true
    ///
    /// - e.g. For negative and positive inputs,
    /// (-3, 3)->(101, 011)-FlipMSB->(001, 111)-unsignedGreaterThan-> false
    ///
    /// - e.g. For both negative inputs,
    /// (-3 > -4) -> (101>100) -flipMSB-> (001>000)-unsignedGreaterThan-> true
    ///
    /// - Once the MSB bit is flipped, and reattached, unsigned operations can be
    /// done on signed, MSB flipped inputs to enable signed comparisons
    ///
    /// - As an effort to reduce the array_to_vector() and vector_to_array()
    /// transformations, this function does not convert from vector to array as that
    /// operation would be reversed in the next part of the code - Optimization
    /// pass should be able to handle this
    fn get_bin_vec_w_flipped_msb(&self, ip: Node) -> Result<Node> {
        let ip_type = ip.get_type()?;
        let (ip_shape, sc_t) = match &ip_type {
            Type::Array(shape_v, sc_t) => {
                if shape_v.is_empty() {
                    return Err(runtime_error!(
                        "Input argument for custom operation - comparison - has no shape"
                    ));
                }
                if *sc_t != BIT {
                    return Err(runtime_error!(
                        "Input argument for custom operation - comparison - has no BIT ScalarType"
                    ));
                }
                (shape_v, sc_t.clone())
            }
            _ => {
                return Err(runtime_error!(
                    "Input argument for custom operation - comparison - is not Array Type"
                ))
            }
        };

        let graph = ip.get_graph();
        let mut msb_slice: Slice = vec![];
        let mut magnitude_slice: Slice = vec![];
        msb_slice.push(SliceElement::Ellipsis);
        magnitude_slice.push(SliceElement::Ellipsis);
        msb_slice.push(SliceElement::SubArray(Some(-1), None, None));
        magnitude_slice.push(SliceElement::SubArray(None, Some(-1), Some(1)));
        let sign_bit = graph.get_slice(ip.clone(), msb_slice)?;
        let magnitude_bits = graph.get_slice(ip, magnitude_slice)?;
        let flipped_bit = graph.custom_op(CustomOperation::new(Not {}), vec![sign_bit])?;
        // 1. Convert to vector form with each component representing each magnitude bit
        // 2. For the output vector to be returned, determine its individual
        // component type. Notes:
        //  2.1 Input was an array (Type::Array) with single axis or multiple axes
        //  2.2 Number of bits would be constant in input and output after all the
        // slicing, negation, array-to-vector and combining/stitching operations
        let magnitude_vec: Node;
        let op_vec_type: Type;
        // total_bits <- #magnitude_bits + #sign_bit(1)
        let total_bits = ip_shape[ip_shape.len() - 1];
        let ip_num_axes = ip_shape.len();
        if ip_num_axes == 1 {
            magnitude_vec = graph.array_to_vector(magnitude_bits)?;
            op_vec_type = vector_type(total_bits, scalar_type(sc_t));
        } else {
            magnitude_vec = self.get_bin_vec(magnitude_bits)?;
            // The innermost dimension of the input array corresponds to the bits and
            // after the conversion to vector, each component is now an array of shape
            // ip_shape[0..(ip_shape.len()-1)]
            op_vec_type = vector_type(
                total_bits,
                array_type(ip_shape[0..(ip_num_axes - 1)].to_vec(), sc_t),
            );
        };
        // Combine the magnitude and sign bits back after flipping the latter i.e.
        // sign bit
        let combined_tup = graph.create_tuple(vec![magnitude_vec, flipped_bit])?;
        let combined_vec = graph.reshape(combined_tup, op_vec_type)?;

        Ok(combined_vec)
    }
}

/// - Certain custom operations are the building-blocks for other similar
/// comparison operations
/// - e.g. GreaterThan, a 'primitive' comparison operation,
/// can be used to build other comparison operations such as LessThan,
/// LessThanEqualTo, GreaterThanEqualTo
/// - Likewise, for NotEqual being a primitive operation for Equal custom
/// comparison operation
/// - Thus, we define a trait `PrimitiveComparisonCustomOperation` for such primitive
/// operations
/// - The trait specifies that there could be graph operating per bit given
/// by get_bit_lvl_graph(), which these operations could define individually
/// - The other trait functionality could be the actual construction of the
/// respective graph, whether GreaterThan or NotEqual, using this bit-level,
/// low granular graph given by `get_bit_lvl_graph()`
#[typetag::serde(tag = "type")]
trait PrimitiveComparisonCustomOperation: ComparisonCustomOperation {
    /// - Depending on the custom comparison operation being designed,
    /// this function should return a low-granular graph operating on
    /// two input bits and a state bit
    /// - Primitive comparison operations such as GreaterThan or NotEqual
    /// should provide this behavior
    /// - The resulting graph could be used as part of an `iterate` graph
    /// operation
    fn get_bit_lvl_graph(
        &self,
        context: Context,
        constant_type: Type,
        ip_a_array_type: Type,
        ip_b_array_type: Type,
    ) -> Result<Graph>;

    /// - This function varies if signed comparison is to be made or an unsigned
    /// comparison has to be made
    /// - Signed comparisons typically would involve slicing, flipping and
    /// repositioning the MSB bit, permuting the axes and then converting from
    /// array to vector types. See implementation for GreaterThan custom op for
    /// more details
    /// - Unsigned comparisons would involve just permuting the axes and then
    /// converting from array to vector types. See implementation for
    /// GreaterThan and NotEqual for more details
    fn preprocess_inputs(&self, ip_a: Node, ip_b: Node) -> Result<(Node, Node)>;

    /// This is a generic function and returns graphs for the basic building blocks of
    /// the custom operations, namely, GreaterThan {} and NotEqual {}.
    /// This functions handles pre-processing of input types to support vectorized inputs.
    fn create_comparison_custom_op(
        &self,
        context: Context,
        arguments_types: Vec<Type>,
    ) -> Result<Graph> {
        self.validate_comparison_arguments(arguments_types.clone())?;

        // Pre-processing steps for both the arguments
        let (bit_vect_len0, new_shape0, new_array_type0, mult_dim_op0) =
            preprocess_comparison_args(arguments_types[0].clone())?;
        let (bit_vect_len1, new_shape1, new_array_type1, mult_dim_op1) =
            preprocess_comparison_args(arguments_types[1].clone())?;

        let constant_type = if mult_dim_op0 || mult_dim_op1 {
            let constant_shape = broadcast_shapes(new_shape0, new_shape1)?;
            array_type(constant_shape, BIT)
        } else {
            scalar_type(BIT)
        };

        let bit_level_comparison_graph = self.get_bit_lvl_graph(
            context.clone(),
            constant_type.clone(),
            new_array_type0.clone(),
            new_array_type1.clone(),
        )?;

        let graph_comp_n_bits = context.create_graph()?;
        {
            // Graph to compare two arrays/scalars or a combination
            // of the two in their vector forms of lengths (l) bit_vect_len0 and
            // bit_vect_len1. Each vector component represents the corresponding
            // bit location, i.e. component 0 (l-1) value -> LSB (MSB) bit value.
            // Assumption: bit_vect_len0 == bit_vect_len1
            let inputs = graph_comp_n_bits.input(tuple_type(vec![
                vector_type(bit_vect_len0, new_array_type0),
                vector_type(bit_vect_len1, new_array_type1),
            ]))?;

            let a = graph_comp_n_bits.tuple_get(inputs.clone(), 0)?;
            let b = graph_comp_n_bits.tuple_get(inputs, 1)?;
            let azb = graph_comp_n_bits.zip(vec![a, b])?;

            let prev_r = zeros(&graph_comp_n_bits, constant_type)?;

            let r_tuple = graph_comp_n_bits.iterate(bit_level_comparison_graph, prev_r, azb)?;
            let r = graph_comp_n_bits.tuple_get(r_tuple, 0)?;
            graph_comp_n_bits.set_output_node(r)?;
            graph_comp_n_bits.finalize()?;
        }

        // Graph to compare two unsigned numbers as represented as ArrayType and
        // having the same final/innermost dimension length
        let comparison_custom_op_graph = context.create_graph()?;
        let a = comparison_custom_op_graph.input(arguments_types[0].clone())?;
        let b = comparison_custom_op_graph.input(arguments_types[1].clone())?;

        // If signed operation, flip MSB bits and then do the `permute_axes()` and `array_to_vector()`
        // else for NotEqual, Equal or unsigned comparisons, just do latter two and avoid flipping MSB part
        let (a_vec_bin, b_vec_bin) = self.preprocess_inputs(a, b)?;

        let arg_comp_64b = comparison_custom_op_graph.create_tuple(vec![a_vec_bin, b_vec_bin])?;
        let result = comparison_custom_op_graph.call(graph_comp_n_bits, vec![arg_comp_64b])?;
        comparison_custom_op_graph.set_output_node(result)?;
        comparison_custom_op_graph.finalize()?;
        Ok(comparison_custom_op_graph)
    }
}

/// A structure that defines the custom operation GreaterThan that compares arrays of binary strings elementwise as follows:
///
/// If a and b are two bitstrings, then GreaterThan(a,b) = 1 if a > b and 0 otherwise.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2]`.
///
/// To compare signed numbers, `signed_comparison` should be set `true`.
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
/// New GreaterThan node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::comparisons::GreaterThan;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(GreaterThan {signed_comparison: false}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct GreaterThan {
    /// Boolean value indicating whether input bitstring represent signed integers
    pub signed_comparison: bool,
}

#[typetag::serde]
impl CustomOperationBody for GreaterThan {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        self.create_comparison_custom_op(context, arguments_types)
    }

    fn get_name(&self) -> String {
        format!("GreaterThan(signed_comparison={})", self.signed_comparison)
    }
}

#[typetag::serde]
impl ComparisonCustomOperation for GreaterThan {
    fn validate_c_op_specific_arguments(&self, arguments_types: Vec<Type>) -> Result<()> {
        self.validate_signed_arguments(arguments_types)
    }
}

#[typetag::serde]
impl NeedsSignedPreprocessing for GreaterThan {
    fn is_signed_custom_operation(&self) -> bool {
        self.signed_comparison
    }
}

#[typetag::serde]
impl PrimitiveComparisonCustomOperation for GreaterThan {
    /// Graph to compare a single bit: (a > b)?
    /// The logic assumes traversing from LSB to MSB
    /// For current pair of bits, for two inputs to be compared,
    /// #1 if current bits differ in a favorable way s.t. a_i > b_i then propagate
    ///    the result ahead, else propagate previous result or unfavorable result ahead
    /// #2 if current bits differ then stop the propagation of result generated
    ///    for previous lower significant bits, else let previous result be propagated ahead
    /// Formula: `cur_r = (a ^ prev_r) & (b ^ prev_r) ^ a`
    fn get_bit_lvl_graph(
        &self,
        context: Context,
        constant_type: Type,
        ip_a_array_type: Type,
        ip_b_array_type: Type,
    ) -> Result<Graph> {
        let is_greater_than_bit_graph = context.create_graph()?;
        let input_prev_r = is_greater_than_bit_graph.input(constant_type)?;
        let inputs =
            is_greater_than_bit_graph.input(tuple_type(vec![ip_a_array_type, ip_b_array_type]))?;
        let a = is_greater_than_bit_graph.tuple_get(inputs.clone(), 0)?;
        let b = is_greater_than_bit_graph.tuple_get(inputs, 1)?;
        let a_xor_prev_r = a.add(input_prev_r.clone())?;
        let b_xor_prev_r = b.add(input_prev_r)?;
        let output_r = a_xor_prev_r.multiply(b_xor_prev_r)?.add(a)?;
        let empty = is_greater_than_bit_graph.create_tuple(vec![])?;
        let output_tuple = is_greater_than_bit_graph.create_tuple(vec![output_r, empty])?;
        is_greater_than_bit_graph.set_output_node(output_tuple)?;
        is_greater_than_bit_graph.add_annotation(GraphAnnotation::OneBitState)?;
        is_greater_than_bit_graph.finalize()?;
        Ok(is_greater_than_bit_graph)
    }

    /// Behavior varies if operation is signed or unsigned as explained in the
    /// trait declaration of this function
    fn preprocess_inputs(&self, ip_a: Node, ip_b: Node) -> Result<(Node, Node)> {
        let (a_vec_bin, b_vec_bin) = if self.signed_comparison {
            // - If the custom comparison operation is signed, flip the bits and do
            // rest of the pre-processing computations of `permute_axes()`
            // (if required) and `array_to_vector()`
            // - Using default method from trait `NeedsSignedPreprocessing`
            (
                self.get_bin_vec_w_flipped_msb(ip_a)?,
                self.get_bin_vec_w_flipped_msb(ip_b)?,
            )
        } else {
            // - Permute and generate vectors to leverage broadcasting rules for input a, b
            // - Using default method from trait `ComparisonCustomOperation` as there are
            // no special requirements to flip the bit
            (self.get_bin_vec(ip_a)?, self.get_bin_vec(ip_b)?)
        };
        Ok((a_vec_bin, b_vec_bin))
    }
}

/// A structure that defines the custom operation NotEqual that compares arrays of binary strings elementwise as follows:
///
/// If a and b are two bitstrings, then NotEqual(a,b) = 1 if a != b and 0 otherwise.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2]`.
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
/// New NotEqual node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::comparisons::NotEqual;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(NotEqual {}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct NotEqual {}

#[typetag::serde]
impl CustomOperationBody for NotEqual {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        self.create_comparison_custom_op(context, arguments_types)
    }

    fn get_name(&self) -> String {
        "NotEqual".to_owned()
    }
}

#[typetag::serde]
impl ComparisonCustomOperation for NotEqual {}

#[typetag::serde]
impl PrimitiveComparisonCustomOperation for NotEqual {
    /// - Returns 1 if input_prev_r == 1 OR tuple inputs = {(1, 0), (0, 1)}, else returns 0.
    /// - This function generates a Graph to evaluate if input two bits given are not equal or to
    /// propagate the previous state result if it is 1.
    /// - The logic assumes traversing from LSB to MSB
    /// - There are two inputs - tuple of bits 'a' and 'b', which are to be tested for non-equality
    /// and previous_bits' state resulting out of use of this function.
    /// - To obtain result, XOR current pair of bits in the tuple and OR the outcome with input_prev_state.
    /// - Obtain XOR by using Add() Graph Operation given inputs' `BIT` `ScalarType` and OR by using the OR custom op.
    /// Formula: `output_r = (a ^ b) | input_prev_r`
    fn get_bit_lvl_graph(
        &self,
        context: Context,
        constant_type: Type,
        ip_a_array_type: Type,
        ip_b_array_type: Type,
    ) -> Result<Graph> {
        let are_not_equal_bit_graph = context.create_graph()?;
        let input_prev_r = are_not_equal_bit_graph.input(constant_type)?;
        let inputs =
            are_not_equal_bit_graph.input(tuple_type(vec![ip_a_array_type, ip_b_array_type]))?;
        let a = are_not_equal_bit_graph.tuple_get(inputs.clone(), 0)?;
        let b = are_not_equal_bit_graph.tuple_get(inputs, 1)?;
        let diff_bit = are_not_equal_bit_graph.add(a, b)?;
        let output_r = are_not_equal_bit_graph
            .custom_op(CustomOperation::new(Or {}), vec![input_prev_r, diff_bit])?;
        let empty = are_not_equal_bit_graph.create_tuple(vec![])?;
        let output_tuple = are_not_equal_bit_graph.create_tuple(vec![output_r, empty])?;
        are_not_equal_bit_graph.set_output_node(output_tuple)?;
        are_not_equal_bit_graph.add_annotation(GraphAnnotation::OneBitState)?;
        are_not_equal_bit_graph.finalize()?;
        Ok(are_not_equal_bit_graph)
    }

    /// - Behavior varies if operation is signed or unsigned as explained in the
    /// trait declaration of this function
    /// - NotEqual and Equal are only unsigned
    fn preprocess_inputs(&self, ip_a: Node, ip_b: Node) -> Result<(Node, Node)> {
        // - Permute and generate vectors to leverage broadcasting rules for input a, b
        // - Using default method from trait `ComparisonCustomOperation` as there are
        // no special requirements to flip the bit, as this is not signed supported
        // operation
        let (a_vec_bin, b_vec_bin) = (self.get_bin_vec(ip_a)?, self.get_bin_vec(ip_b)?);
        Ok((a_vec_bin, b_vec_bin))
    }
}

/// A structure that defines the custom operation LessThan that compares arrays of binary strings elementwise as follows:
///
/// If a and b are two bitstrings, then LessThan(a,b) = 1 if a < b and 0 otherwise.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2]`.
///
/// To compare signed numbers, `signed_comparison` should be set `true`.
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
/// New LessThan node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::comparisons::LessThan;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(LessThan {signed_comparison: true}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct LessThan {
    /// Boolean value indicating whether input bitstring represent signed integers
    pub signed_comparison: bool,
}

#[typetag::serde]
impl CustomOperationBody for LessThan {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        self.validate_comparison_arguments(arguments_types.clone())?;
        //Graph to compare two n bit numbers as arrays
        let less_than_graph = context.create_graph()?;
        let a = less_than_graph.input(arguments_types[0].clone())?;
        let b = less_than_graph.input(arguments_types[1].clone())?;
        let result = less_than_graph.custom_op(
            CustomOperation::new(GreaterThan {
                signed_comparison: self.signed_comparison,
            }),
            vec![b, a],
        )?;
        less_than_graph.set_output_node(result)?;
        less_than_graph.finalize()?;
        Ok(less_than_graph)
    }

    fn get_name(&self) -> String {
        format!("LessThan(signed_comparison={})", self.signed_comparison)
    }
}

#[typetag::serde]
impl ComparisonCustomOperation for LessThan {
    fn validate_c_op_specific_arguments(&self, arguments_types: Vec<Type>) -> Result<()> {
        self.validate_signed_arguments(arguments_types)
    }
}

#[typetag::serde]
impl NeedsSignedPreprocessing for LessThan {
    fn is_signed_custom_operation(&self) -> bool {
        self.signed_comparison
    }
}

/// A structure that defines the custom operation LessThanEqualTo that compares arrays of binary strings elementwise as follows:
///
/// If a and b are two bitstrings, then LessThanEqualTo(a,b) = 1 if a <= b and 0 otherwise.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2]`.
///
/// To compare signed numbers, `signed_comparison` should be set `true`.
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
/// New LessThanEqualTo node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::comparisons::LessThanEqualTo;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(LessThanEqualTo {signed_comparison: true}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct LessThanEqualTo {
    /// Boolean value indicating whether input bitstring represent signed integers
    pub signed_comparison: bool,
}

#[typetag::serde]
impl CustomOperationBody for LessThanEqualTo {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        self.validate_comparison_arguments(arguments_types.clone())?;
        let less_than_equal_to = context.create_graph()?;
        let a = less_than_equal_to.input(arguments_types[0].clone())?;
        let b = less_than_equal_to.input(arguments_types[1].clone())?;
        let a_gt_b = less_than_equal_to.custom_op(
            CustomOperation::new(GreaterThan {
                signed_comparison: self.signed_comparison,
            }),
            vec![a, b],
        )?;
        let result = less_than_equal_to.custom_op(CustomOperation::new(Not {}), vec![a_gt_b])?;
        less_than_equal_to.set_output_node(result)?;
        less_than_equal_to.finalize()?;
        Ok(less_than_equal_to)
    }

    fn get_name(&self) -> String {
        format!(
            "LessThanEqualTo(signed_comparison={})",
            self.signed_comparison
        )
    }
}

#[typetag::serde]
impl ComparisonCustomOperation for LessThanEqualTo {
    fn validate_c_op_specific_arguments(&self, arguments_types: Vec<Type>) -> Result<()> {
        self.validate_signed_arguments(arguments_types)
    }
}

#[typetag::serde]
impl NeedsSignedPreprocessing for LessThanEqualTo {
    fn is_signed_custom_operation(&self) -> bool {
        self.signed_comparison
    }
}

/// A structure that defines the custom operation GreaterThanEqualTo that compares arrays of binary strings elementwise as follows:
///
/// If a and b are two bitstrings, then GreaterThanEqualTo(a,b) = 1 if a >= b and 0 otherwise.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2]`.
///
/// To compare signed numbers, `signed_comparison` should be set `true`.
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
/// New GreaterThanEqualTo node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::comparisons::GreaterThanEqualTo;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(GreaterThanEqualTo {signed_comparison: true}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct GreaterThanEqualTo {
    /// Boolean value indicating whether input bitstring represent signed integers
    pub signed_comparison: bool,
}

#[typetag::serde]
impl CustomOperationBody for GreaterThanEqualTo {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        self.validate_comparison_arguments(arguments_types.clone())?;
        let greater_than_equal_to = context.create_graph()?;
        let a = greater_than_equal_to.input(arguments_types[0].clone())?;
        let b = greater_than_equal_to.input(arguments_types[1].clone())?;
        let a_lt_b = greater_than_equal_to.custom_op(
            CustomOperation::new(LessThan {
                signed_comparison: self.signed_comparison,
            }),
            vec![a, b],
        )?;
        let result = greater_than_equal_to.custom_op(CustomOperation::new(Not {}), vec![a_lt_b])?;
        greater_than_equal_to.set_output_node(result)?;
        greater_than_equal_to.finalize()?;
        Ok(greater_than_equal_to)
    }

    fn get_name(&self) -> String {
        format!(
            "GreaterThanEqualTo(signed_comparison={})",
            self.signed_comparison
        )
    }
}

#[typetag::serde]
impl ComparisonCustomOperation for GreaterThanEqualTo {
    fn validate_c_op_specific_arguments(&self, arguments_types: Vec<Type>) -> Result<()> {
        self.validate_signed_arguments(arguments_types)
    }
}

#[typetag::serde]
impl NeedsSignedPreprocessing for GreaterThanEqualTo {
    fn is_signed_custom_operation(&self) -> bool {
        self.signed_comparison
    }
}

/// A structure that defines the custom operation Equal that compares arrays of binary strings elementwise as follows:
///
/// If a and b are two bitstrings, then Equal(a,b) = 1 if a = b and 0 otherwise.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2]`.
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
/// New Equal node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::comparisons::Equal;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 3], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(Equal {}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Equal {}

#[typetag::serde]
impl CustomOperationBody for Equal {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        self.validate_comparison_arguments(arguments_types.clone())?;
        //Graph to compare two n bit numbers as arrays
        let equal_to = context.create_graph()?;
        let a = equal_to.input(arguments_types[0].clone())?;
        let b = equal_to.input(arguments_types[1].clone())?;
        let is_not_equal = equal_to.custom_op(CustomOperation::new(NotEqual {}), vec![a, b])?;
        let result = equal_to.custom_op(CustomOperation::new(Not {}), vec![is_not_equal])?;
        equal_to.set_output_node(result)?;
        equal_to.finalize()?;
        Ok(equal_to)
    }

    fn get_name(&self) -> String {
        "Equal".to_owned()
    }
}

#[typetag::serde]
impl ComparisonCustomOperation for Equal {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::{
        array_type, ScalarType, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8,
    };
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;

    fn test_unsigned_greater_than_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(GreaterThan {
                signed_comparison: false,
            }),
            vec![i_a, i_b],
        )?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    fn test_signed_greater_than_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(GreaterThan {
                signed_comparison: true,
            }),
            vec![i_a, i_b],
        )?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        let random_val = random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?;
        let op = random_val.to_u8(BIT)?;
        Ok(op)
    }

    fn test_not_equal_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(CustomOperation::new(NotEqual {}), vec![i_a, i_b])?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    /// Given supported bit size returns unsigned ScalarType
    fn get_u_scalar_type_from_bits(bit_size: u64) -> Result<ScalarType> {
        match bit_size {
            8 => Ok(UINT8),
            16 => Ok(UINT16),
            32 => Ok(UINT32),
            64 => Ok(UINT64),
            _ => Err(runtime_error!("Unsupported bit size")),
        }
    }

    /// Given supported bit size returns signed ScalarType
    fn get_s_scalar_type_from_bits(bit_size: u64) -> Result<ScalarType> {
        match bit_size {
            8 => Ok(INT8),
            16 => Ok(INT16),
            32 => Ok(INT32),
            64 => Ok(INT64),
            _ => Err(runtime_error!("Unsupported bit size")),
        }
    }

    /// Inputs:
    /// comparison_op
    /// a: input assumed to pass as Vec<64>
    /// b: input assumed to pass as Vec<64>
    /// shape_a: intended shape for a within graph
    /// shape_b: intended shape for b within graph
    fn test_unsigned_comparison_cust_op_for_vec_helper(
        comparison_op: CustomOperation,
        a: Vec<u64>,
        b: Vec<u64>,
        shape_a: ArrayShape,
        shape_b: ArrayShape,
    ) -> Result<Vec<u64>> {
        let bit_vector_len_a = shape_a[shape_a.len() - 1];
        let bit_vector_len_b = shape_b[shape_b.len() - 1];
        let data_scalar_type_a = get_u_scalar_type_from_bits(bit_vector_len_a)?;
        let data_scalar_type_b = get_u_scalar_type_from_bits(bit_vector_len_b)?;

        let c = create_context()?;
        let g = c.create_graph()?;
        let i_va = g.input(array_type(shape_a.clone(), BIT))?;
        let i_vb = g.input(array_type(shape_b.clone(), BIT))?;
        let o = g.custom_op(comparison_op.clone(), vec![i_va, i_vb])?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        // Restructure the input data
        let v_a = Value::from_flattened_array(&a, data_scalar_type_a)?;
        let v_b = Value::from_flattened_array(&b, data_scalar_type_b)?;
        let broadcasted_output_shape = broadcast_shapes(
            shape_a[0..(shape_a.len() - 1)].to_vec(),
            shape_b[0..(shape_b.len() - 1)].to_vec(),
        )?;

        let result = random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?
            .to_flattened_array_u64(array_type(broadcasted_output_shape, BIT))?;
        Ok(result)
    }

    /// Inputs:
    /// comparison_op
    /// a: input assumed to pass as Vec<64>
    /// b: input assumed to pass as Vec<64>
    /// shape_a: intended shape for a within graph
    /// shape_b: intended shape for b within graph
    fn test_signed_comparison_cust_op_for_vec_helper(
        comparison_op: CustomOperation,
        a: Vec<i64>,
        b: Vec<i64>,
        shape_a: ArrayShape,
        shape_b: ArrayShape,
    ) -> Result<Vec<u64>> {
        let bit_vector_len_a = shape_a[shape_a.len() - 1];
        let bit_vector_len_b = shape_b[shape_b.len() - 1];
        let data_scalar_type_a = get_s_scalar_type_from_bits(bit_vector_len_a)?;
        let data_scalar_type_b = get_s_scalar_type_from_bits(bit_vector_len_b)?;

        let c = create_context()?;
        let g = c.create_graph()?;
        let i_va = g.input(array_type(shape_a.clone(), BIT))?;
        let i_vb = g.input(array_type(shape_b.clone(), BIT))?;
        let o = g.custom_op(comparison_op.clone(), vec![i_va, i_vb])?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        // Restructure the input data
        let v_a = Value::from_flattened_array(&a, data_scalar_type_a)?;
        let v_b = Value::from_flattened_array(&b, data_scalar_type_b)?;
        let broadcasted_output_shape = broadcast_shapes(
            shape_a[0..(shape_a.len() - 1)].to_vec(),
            shape_b[0..(shape_b.len() - 1)].to_vec(),
        )?;

        let result = random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?
            .to_flattened_array_u64(array_type(broadcasted_output_shape, BIT))?;
        Ok(result)
    }

    fn test_unsigned_less_than_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(LessThan {
                signed_comparison: false,
            }),
            vec![i_a, i_b],
        )?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    fn test_signed_less_than_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(LessThan {
                signed_comparison: true,
            }),
            vec![i_a, i_b],
        )?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    fn test_unsigned_less_than_equal_to_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(LessThanEqualTo {
                signed_comparison: false,
            }),
            vec![i_a, i_b],
        )?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    fn test_signed_less_than_equal_to_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(LessThanEqualTo {
                signed_comparison: true,
            }),
            vec![i_a, i_b],
        )?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    fn test_unsigned_greater_than_equal_to_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(GreaterThanEqualTo {
                signed_comparison: false,
            }),
            vec![i_a, i_b],
        )?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    fn test_signed_greater_than_equal_to_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(GreaterThanEqualTo {
                signed_comparison: true,
            }),
            vec![i_a, i_b],
        )?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    fn test_equal_to_cust_op_helper(a: Vec<u8>, b: Vec<u8>) -> Result<u8> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i_a = g.input(array_type(vec![a.len() as u64], BIT))?;
        let i_b = g.input(array_type(vec![b.len() as u64], BIT))?;
        let o = g.custom_op(CustomOperation::new(Equal {}), vec![i_a, i_b])?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;
        let v_a = Value::from_flattened_array(&a, BIT)?;
        let v_b = Value::from_flattened_array(&b, BIT)?;
        Ok(random_evaluate(mapped_c.mappings.get_graph(g), vec![v_a, v_b])?.to_u8(BIT)?)
    }

    #[test]
    fn test_greater_than_cust_op() {
        || -> Result<()> {
            assert_eq!(
                test_unsigned_greater_than_cust_op_helper(vec![0], vec![0])?,
                0
            );
            assert_eq!(
                test_unsigned_greater_than_cust_op_helper(vec![0], vec![1])?,
                0
            );
            assert_eq!(
                test_unsigned_greater_than_cust_op_helper(vec![1], vec![0])?,
                1
            );
            assert_eq!(
                test_unsigned_greater_than_cust_op_helper(vec![1], vec![1])?,
                0
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_signed_greater_than_cust_op() {
        || -> Result<()> {
            // for signed positive values
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![0, 0], vec![0, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![0, 0], vec![1, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![1, 0], vec![0, 0])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![1, 0], vec![1, 0])?,
                0
            );
            // for signed negative values
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![0, 1], vec![0, 1])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![0, 1], vec![1, 1])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![1, 1], vec![0, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![1, 1], vec![1, 1])?,
                0
            );
            // mixture of values
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![0, 1], vec![0, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![0, 0], vec![0, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![0, 1], vec![1, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![0, 0], vec![1, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![1, 1], vec![0, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![1, 0], vec![0, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![1, 1], vec![1, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_cust_op_helper(vec![1, 0], vec![1, 1])?,
                1
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_unsigned_less_than_cust_op() {
        || -> Result<()> {
            assert_eq!(test_unsigned_less_than_cust_op_helper(vec![0], vec![0])?, 0);
            assert_eq!(test_unsigned_less_than_cust_op_helper(vec![0], vec![1])?, 1);
            assert_eq!(test_unsigned_less_than_cust_op_helper(vec![1], vec![0])?, 0);
            assert_eq!(test_unsigned_less_than_cust_op_helper(vec![1], vec![1])?, 0);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_signed_less_than_cust_op() {
        || -> Result<()> {
            // for signed positive values
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![0, 0], vec![0, 0])?,
                0
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![0, 0], vec![1, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![1, 0], vec![0, 0])?,
                0
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![1, 0], vec![1, 0])?,
                0
            );
            // for signed negative less
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![0, 1], vec![0, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![0, 1], vec![1, 1])?,
                1
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![1, 1], vec![0, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![1, 1], vec![1, 1])?,
                0
            );
            // mixture of valuesless
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![0, 1], vec![0, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![0, 0], vec![0, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![0, 1], vec![1, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![0, 0], vec![1, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![1, 1], vec![0, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![1, 0], vec![0, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![1, 1], vec![1, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_cust_op_helper(vec![1, 0], vec![1, 1])?,
                0
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_unsigned_less_than_or_eq_to_cust_op() {
        || -> Result<()> {
            assert_eq!(
                test_unsigned_less_than_equal_to_cust_op_helper(vec![0], vec![0])?,
                1
            );
            assert_eq!(
                test_unsigned_less_than_equal_to_cust_op_helper(vec![0], vec![1])?,
                1
            );
            assert_eq!(
                test_unsigned_less_than_equal_to_cust_op_helper(vec![1], vec![0])?,
                0
            );
            assert_eq!(
                test_unsigned_less_than_equal_to_cust_op_helper(vec![1], vec![1])?,
                1
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_signed_less_than_or_eq_to_cust_op() {
        || -> Result<()> {
            // for signed positive values
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![0, 0], vec![0, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![0, 0], vec![1, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![1, 0], vec![0, 0])?,
                0
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![1, 0], vec![1, 0])?,
                1
            );
            // for signed negative less
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![0, 1], vec![0, 1])?,
                1
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![0, 1], vec![1, 1])?,
                1
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![1, 1], vec![0, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![1, 1], vec![1, 1])?,
                1
            );
            // mixture of valuesless
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![0, 1], vec![0, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![0, 0], vec![0, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![0, 1], vec![1, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![0, 0], vec![1, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![1, 1], vec![0, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![1, 0], vec![0, 1])?,
                0
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![1, 1], vec![1, 0])?,
                1
            );
            assert_eq!(
                test_signed_less_than_equal_to_cust_op_helper(vec![1, 0], vec![1, 1])?,
                0
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_unsigned_greater_than_or_eq_to_cust_op() {
        || -> Result<()> {
            assert_eq!(
                test_unsigned_greater_than_equal_to_cust_op_helper(vec![0], vec![0])?,
                1
            );
            assert_eq!(
                test_unsigned_greater_than_equal_to_cust_op_helper(vec![0], vec![1])?,
                0
            );
            assert_eq!(
                test_unsigned_greater_than_equal_to_cust_op_helper(vec![1], vec![0])?,
                1
            );
            assert_eq!(
                test_unsigned_greater_than_equal_to_cust_op_helper(vec![1], vec![1])?,
                1
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_signed_greater_than_or_eq_to_cust_op() {
        || -> Result<()> {
            // for signed positive values
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![0, 0], vec![0, 0])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![0, 0], vec![1, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![1, 0], vec![0, 0])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![1, 0], vec![1, 0])?,
                1
            );
            // for signed negative values
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![0, 1], vec![0, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![0, 1], vec![1, 1])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![1, 1], vec![0, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![1, 1], vec![1, 1])?,
                1
            );
            // mixture of values
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![0, 1], vec![0, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![0, 0], vec![0, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![0, 1], vec![1, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![0, 0], vec![1, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![1, 1], vec![0, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![1, 0], vec![0, 1])?,
                1
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![1, 1], vec![1, 0])?,
                0
            );
            assert_eq!(
                test_signed_greater_than_equal_to_cust_op_helper(vec![1, 0], vec![1, 1])?,
                1
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_not_equal_cust_op() {
        || -> Result<()> {
            assert_eq!(test_not_equal_cust_op_helper(vec![0], vec![0])?, 0);
            assert_eq!(test_not_equal_cust_op_helper(vec![0], vec![1])?, 1);
            assert_eq!(test_not_equal_cust_op_helper(vec![1], vec![0])?, 1);
            assert_eq!(test_not_equal_cust_op_helper(vec![1], vec![1])?, 0);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_equal_to_cust_op() {
        || -> Result<()> {
            assert_eq!(test_equal_to_cust_op_helper(vec![0], vec![0])?, 1);
            assert_eq!(test_equal_to_cust_op_helper(vec![0], vec![1])?, 0);
            assert_eq!(test_equal_to_cust_op_helper(vec![1], vec![0])?, 0);
            assert_eq!(test_equal_to_cust_op_helper(vec![1], vec![1])?, 1);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_unsigned_multiple_bit_comparisons_cust_op() {
        || -> Result<()> {
            for i in 0..8 {
                for j in 0..8 {
                    let a: Vec<u8> = vec![i & 1, (i & 2) >> 1, (i & 4) >> 2];
                    let b: Vec<u8> = vec![j & 1, (j & 2) >> 1, (j & 4) >> 2];
                    assert_eq!(
                        test_unsigned_greater_than_cust_op_helper(a.clone(), b.clone())?,
                        if i > j { 1 } else { 0 }
                    );
                    assert_eq!(
                        test_unsigned_less_than_cust_op_helper(a.clone(), b.clone())?,
                        if i < j { 1 } else { 0 }
                    );
                    assert_eq!(
                        test_unsigned_greater_than_equal_to_cust_op_helper(a.clone(), b.clone())?,
                        if i >= j { 1 } else { 0 }
                    );
                    assert_eq!(
                        test_unsigned_less_than_equal_to_cust_op_helper(a.clone(), b.clone())?,
                        if i <= j { 1 } else { 0 }
                    );
                    assert_eq!(
                        test_not_equal_cust_op_helper(a.clone(), b.clone())?,
                        if i != j { 1 } else { 0 }
                    );
                    assert_eq!(
                        test_equal_to_cust_op_helper(a.clone(), b.clone())?,
                        if i == j { 1 } else { 0 }
                    );
                }
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_signed_multiple_bit_comparisons_cust_op() {
        || -> Result<()> {
            for i in 0..8 {
                for j in 0..8 {
                    let a: Vec<u8> = vec![i & 1, (i & 2) >> 1, (i & 4) >> 2];
                    let b: Vec<u8> = vec![j & 1, (j & 2) >> 1, (j & 4) >> 2];
                    let s_i = if i > 3 { i as i8 - 8 } else { i as i8 };
                    let s_j = if j > 3 { j as i8 - 8 } else { j as i8 };
                    assert_eq!(
                        test_signed_greater_than_cust_op_helper(a.clone(), b.clone())?,
                        if s_i > s_j { 1 } else { 0 }
                    );
                    assert_eq!(
                        test_signed_less_than_cust_op_helper(a.clone(), b.clone())?,
                        if s_i < s_j { 1 } else { 0 }
                    );
                    assert_eq!(
                        test_signed_greater_than_equal_to_cust_op_helper(a.clone(), b.clone())?,
                        if s_i >= s_j { 1 } else { 0 }
                    );
                    assert_eq!(
                        test_signed_less_than_equal_to_cust_op_helper(a.clone(), b.clone())?,
                        if s_i <= s_j { 1 } else { 0 }
                    );
                }
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_unsigned_malformed_basic_cust_ops() {
        || -> Result<()> {
            let cust_ops = vec![
                CustomOperation::new(GreaterThan {
                    signed_comparison: false,
                }),
                CustomOperation::new(NotEqual {}),
            ];
            for cust_op in cust_ops.into_iter() {
                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![1], BIT))?;
                let i_c = g.input(array_type(vec![1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b, i_c]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(scalar_type(BIT))?;
                let i_b = g.input(array_type(vec![1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(tuple_type(vec![array_type(vec![1], BIT)]))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], INT16))?;
                let i_b = g.input(array_type(vec![1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], UINT16))?;
                let i_b = g.input(array_type(vec![1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![1], INT32))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![1], UINT32))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![9], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![1, 2], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let v_a = vec![170, 120, 61, 85];
                let v_b = vec![76, 20, 70, 249, 217, 190, 43, 83, 33710];
                assert!(test_unsigned_comparison_cust_op_for_vec_helper(
                    cust_op.clone(),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 16],
                    vec![3, 3, 32]
                )
                .is_err());

                let v_a = vec![170];
                let v_b = vec![76, 20, 70, 249, 217, 190, 43, 83, 33710];
                assert!(test_unsigned_comparison_cust_op_for_vec_helper(
                    cust_op.clone(),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 16],
                    vec![3, 3, 16]
                )
                .is_err());

                let v_a = vec![];
                let v_b = vec![76, 20, 70, 249, 217, 190, 43, 83, 33710];
                assert!(test_unsigned_comparison_cust_op_for_vec_helper(
                    cust_op.clone(),
                    v_a.clone(),
                    v_b.clone(),
                    vec![0, 64],
                    vec![3, 3, 64]
                )
                .is_err());

                let v_a = vec![170, 200];
                let v_b = vec![];
                assert!(test_unsigned_comparison_cust_op_for_vec_helper(
                    cust_op.clone(),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 1, 64],
                    vec![2, 2, 1, 64]
                )
                .is_err());
            }

            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_signed_malformed_basic_cust_ops() {
        || -> Result<()> {
            let cust_ops = vec![CustomOperation::new(GreaterThan {
                signed_comparison: true,
            })];
            for cust_op in cust_ops.into_iter() {
                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![1, 1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1, 1], BIT))?;
                let i_b = g.input(array_type(vec![1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1, 64], BIT))?;
                let i_b = g.input(array_type(vec![1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(scalar_type(BIT))?;
                let i_b = g.input(array_type(vec![1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], UINT16))?;
                let i_b = g.input(array_type(vec![1], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![1], INT32))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![1], UINT32))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![9], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1, 2, 3], BIT))?;
                let i_b = g.input(array_type(vec![9], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let c = create_context()?;
                let g = c.create_graph()?;
                let i_a = g.input(array_type(vec![1], BIT))?;
                let i_b = g.input(array_type(vec![1, 2], BIT))?;
                assert!(g.custom_op(cust_op.clone(), vec![i_a, i_b]).is_err());

                let v_a = vec![170, 120, 61, 85];
                let v_b = vec![
                    -1176658021,
                    -301476304,
                    788180273,
                    -1085188538,
                    -1358798926,
                    -120286105,
                    -1300942710,
                    -389618936,
                    258721418,
                ];
                assert!(test_signed_comparison_cust_op_for_vec_helper(
                    cust_op.clone(),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 16],
                    vec![3, 3, 32]
                )
                .is_err());

                let v_a = vec![-14735];
                let v_b = vec![
                    16490, -10345, -31409, 2787, -15039, 26085, 7881, 32423, -23915,
                ];
                assert!(test_signed_comparison_cust_op_for_vec_helper(
                    cust_op.clone(),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 16],
                    vec![3, 3, 16]
                )
                .is_err());

                let v_a = vec![];
                let v_b = vec![
                    -2600362169875399934,
                    6278463339984150730,
                    -2962726308672949899,
                    3404980137287029349,
                ];
                assert!(test_signed_comparison_cust_op_for_vec_helper(
                    cust_op.clone(),
                    v_a.clone(),
                    v_b.clone(),
                    vec![0, 64],
                    vec![2, 2, 64]
                )
                .is_err());

                let v_a = vec![-2600362169875399934, 6278463339984150730];
                let v_b = vec![];
                assert!(test_signed_comparison_cust_op_for_vec_helper(
                    cust_op.clone(),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 1, 64],
                    vec![2, 2, 1, 64]
                )
                .is_err());
            }

            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_unsigned_vector_comparisons() {
        || -> Result<()> {
            let mut v_a = vec![170, 120, 61, 85];
            let mut v_b = vec![
                76, 20, 70, 249, 217, 190, 43, 83, 33710, 27637, 43918, 38683,
            ];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 64],
                    vec![3, 2, 2, 64],
                )?,
                vec![1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
            );

            v_a = vec![170, 120, 61, 85, 75, 149, 50, 54, 8811, 29720, 1009, 33126];
            v_b = vec![76, 20, 70, 249, 217, 190];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 3, 2, 32],
                    vec![3, 2, 32],
                )?,
                vec![1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            );

            v_a = vec![170, 120, 61, 85, 75, 149, 50, 54];
            v_b = vec![76, 20, 70, 249];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 2, 16],
                    vec![2, 2, 16],
                )?,
                vec![1, 1, 0, 0, 0, 1, 0, 0]
            );

            v_a = vec![170, 120, 61, 85, 75, 149, 50, 54];
            v_b = vec![76, 20, 70, 249, 217, 190, 43, 83];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 2, 64],
                    vec![2, 2, 2, 64],
                )?,
                vec![1, 1, 0, 0, 0, 0, 1, 0]
            );

            v_a = vec![170, 120, 61];
            v_b = vec![76, 20, 70];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![3, 64],
                    vec![3, 64],
                )?,
                vec![1, 1, 0]
            );

            v_a = vec![170, 120, 61, 85, 75, 149];
            v_b = vec![76, 20, 70];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(LessThan {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 3, 64],
                    vec![3, 64],
                )?,
                vec![0, 0, 1, 0, 0, 0]
            );

            v_a = vec![170, 120, 61, 85, 75, 70, 50, 54, 8811, 29720, 1009, 33126];
            v_b = vec![76, 1009, 70];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(LessThanEqualTo {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 3, 64],
                    vec![3, 64],
                )?,
                vec![0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0]
            );

            v_a = vec![170, 120, 61, 85, 75, 76, 50, 54];
            v_b = vec![76];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThanEqualTo {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 2, 64],
                    vec![1, 64],
                )?,
                vec![1, 1, 0, 1, 0, 1, 0, 0]
            );

            v_a = vec![170];
            v_b = vec![76];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThanEqualTo {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![1, 64],
                    vec![1, 64],
                )?,
                vec![1]
            );

            v_a = vec![76];
            v_b = vec![76, 170];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThanEqualTo {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![1, 64],
                    vec![1, 2, 64],
                )?,
                vec![1, 0]
            );

            let v_a = vec![83, 172, 214, 2, 68];
            let v_b = vec![83];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThanEqualTo {
                        signed_comparison: false
                    }),
                    v_a,
                    v_b,
                    vec![5, 8],
                    vec![8]
                )?,
                vec![1, 1, 1, 0, 0]
            );

            let v_a = vec![2];
            let v_b = vec![83, 1, 2, 100];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(LessThan {
                        signed_comparison: false
                    }),
                    v_a,
                    v_b,
                    vec![1, 32],
                    vec![2, 2, 32]
                )?,
                vec![1, 0, 0, 1]
            );

            let v_a = vec![83, 2];
            let v_b = vec![83, 172, 214, 2, 68, 34, 87, 45, 83, 23];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(LessThanEqualTo {
                        signed_comparison: false
                    }),
                    v_a,
                    v_b,
                    vec![2, 1, 64],
                    vec![2, 5, 64]
                )?,
                vec![1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
            );

            let v_a = vec![83, 2];
            let v_b = vec![83, 172, 214, 2, 68, 2, 87, 45];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(NotEqual {}),
                    v_a,
                    v_b,
                    vec![2, 1, 64],
                    vec![2, 4, 64]
                )?,
                vec![0, 1, 1, 1, 1, 0, 1, 1]
            );

            let v_a = vec![4, 2];
            let v_b = vec![83, 21];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(NotEqual {}),
                    v_a,
                    v_b,
                    vec![1, 2, 64],
                    vec![2, 1, 64]
                )?,
                vec![1, 1, 1, 1]
            );

            let v_a = vec![247, 170, 249, 162, 102, 243, 61, 203, 125];
            let v_b = vec![247, 170, 249, 162, 102, 243, 61, 203, 125];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(NotEqual {}),
                    v_a,
                    v_b,
                    vec![3, 3, 16],
                    vec![3, 3, 16]
                )?,
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0]
            );

            let v_a = vec![83, 2];
            let v_b = vec![83, 172, 214, 2, 68, 2, 87, 45];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(Equal {}),
                    v_a,
                    v_b,
                    vec![2, 1, 64],
                    vec![2, 4, 64]
                )?,
                vec![1, 0, 0, 0, 0, 1, 0, 0]
            );

            let v_a = vec![4, 2];
            let v_b = vec![83, 21];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(Equal {}),
                    v_a,
                    v_b,
                    vec![1, 2, 64],
                    vec![2, 1, 64]
                )?,
                vec![0, 0, 0, 0]
            );

            let v_a = vec![180, 16, 62, 141, 122, 217];
            let v_b = vec![141, 122, 217, 100, 11, 29];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(Equal {}),
                    v_a,
                    v_b,
                    vec![3, 2, 1, 16],
                    vec![1, 2, 3, 16]
                )?,
                vec![
                    0, 0, 0, // 180==[141, 122, 217]
                    0, 0, 0, // 16==[100, 11, 29]
                    0, 0, 0, // 62==[141, 122, 217]
                    0, 0, 0, // 141==[100, 11, 29]
                    0, 1, 0, // 122==[141, 122, 217]
                    0, 0, 0 // 217==[100, 11, 29]
                ]
            );

            let v_a = vec![0, 1, 18446744073709551614, 18446744073709551615];
            let v_b = vec![0, 1, 18446744073709551614, 18446744073709551615];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![4, 1, 64],
                    vec![1, 4, 64],
                )?,
                vec![0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
            );

            let v_a = vec![0, 1, 18446744073709551614, 18446744073709551615];
            let v_b = vec![0, 1, 18446744073709551614, 18446744073709551615];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThanEqualTo {
                        signed_comparison: false
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![4, 1, 64],
                    vec![1, 4, 64],
                )?,
                vec![1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
            );

            let v_a = vec![0, 1, 18446744073709551614, 18446744073709551615];
            let v_b = vec![0, 1, 18446744073709551614, 18446744073709551615];
            assert_eq!(
                test_unsigned_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(NotEqual {}),
                    v_a.clone(),
                    v_b.clone(),
                    vec![4, 1, 64],
                    vec![1, 4, 64],
                )?,
                vec![0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
            );

            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_signed_vector_comparisons() {
        || -> Result<()> {
            let v_a = vec![
                -9223372036854775808,
                -9223372036854775807,
                -1,
                0,
                1,
                9223372036854775806,
                9223372036854775807,
            ];
            let v_b = vec![
                -9223372036854775808,
                -9223372036854775807,
                -1,
                0,
                1,
                9223372036854775806,
                9223372036854775807,
            ];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![7, 1, 64],
                    vec![1, 7, 64],
                )?,
                vec![
                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0
                ]
            );

            let v_a = vec![
                -9223372036854775808,
                -9223372036854775807,
                -1,
                0,
                1,
                9223372036854775806,
                9223372036854775807,
            ];
            let v_b = vec![
                -9223372036854775808,
                -9223372036854775807,
                -1,
                0,
                1,
                9223372036854775806,
                9223372036854775807,
            ];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThanEqualTo {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![7, 1, 64],
                    vec![1, 7, 64],
                )?,
                vec![
                    1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
                    0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1
                ]
            );

            let mut v_a = vec![-6749, -1885, 7550, 9659];
            let mut v_b = vec![
                9918, 3462, -5690, 3436, 3214, -1733, 6171, 3148, -3534, 8282, -4904, -5976,
            ];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 64],
                    vec![3, 2, 2, 64],
                )?,
                vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            );

            v_a = vec![
                -48, -9935, -745, 2360, -4597, -5271, 5130, -2632, 3112, 8089, 8293, 6058,
            ];
            v_b = vec![2913, 7260, 1388, 6205, 1855, 3246];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 3, 2, 32],
                    vec![3, 2, 32],
                )?,
                vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
            );

            v_a = vec![9838, -574, -4181, -8107, -2880, -2866, 2272, 3743];
            v_b = vec![626, 4664, 1490, -5118, 7485, 6160, 4221, 2092];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 2, 64],
                    vec![2, 2, 2, 64],
                )?,
                vec![1, 0, 0, 0, 0, 0, 0, 1]
            );

            v_a = vec![-75, 95, -84, 67, -81, 14];
            v_b = vec![-78, 21, -66];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(LessThan {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 3, 8],
                    vec![3, 8],
                )?,
                vec![0, 0, 1, 0, 1, 0]
            );

            v_a = vec![-52, -119, 30, -24, -74, -45, 66, 110, 21, 1, 95, -66];
            v_b = vec![33, -78, 39];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(LessThanEqualTo {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 3, 8],
                    vec![3, 8],
                )?,
                vec![1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
            );

            v_a = vec![-128, 127, 0, 1, 0, -128, 1, 127];
            v_b = vec![-128];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThanEqualTo {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 2, 8],
                    vec![1, 8],
                )?,
                vec![1, 1, 1, 1, 1, 1, 1, 1]
            );

            v_a = vec![-128, 127, 0, 1, 0, -128, 1, 127];
            v_b = vec![-128];
            assert_eq!(
                test_signed_comparison_cust_op_for_vec_helper(
                    CustomOperation::new(GreaterThan {
                        signed_comparison: true
                    }),
                    v_a.clone(),
                    v_b.clone(),
                    vec![2, 2, 2, 8],
                    vec![1, 8],
                )?,
                vec![0, 1, 1, 1, 1, 0, 1, 1]
            );

            Ok(())
        }()
        .unwrap();
    }
}
