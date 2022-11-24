//! Various comparison functions for signed and unsigned integers including greater-than, less-than, greater-than-equal-to, less-than-equal-to, equal, not-equal.
use crate::custom_ops::{CustomOperation, CustomOperationBody, Not};
use crate::data_types::{Type, BIT};
use crate::errors::Result;
use crate::graphs::*;
use crate::ops::utils::validate_arguments_in_broadcast_bit_ops;
use crate::ops::utils::{constant_scalar, pull_out_bits};
use std::cmp::max;

use serde::{Deserialize, Serialize};

/// All comparison operations are built on top of a general comparison graph.
///
/// For each pair of numbers, which are compared, we need to distinguish 3 possible
/// options:
/// - the first number is smaller
/// - numbers are equal
/// - the first number is larger
///
/// To represent it we use 2 bits, which leads to 4 possible options, which doesn't
/// map nicely to 3, so we always have one redundant state.
///
/// There are several possible ways to choose what those two bits represent. We
/// chose the way which, could be a little bit confusing, but which minimizes the
/// number of network rounds needed to compute the final result.
///
/// Let's say we want to compare numbers `a` and `b`. We can find a pair of bits
/// `(a', b')` such that the compare result for `a` and `b` is the same as result
/// for `a'` and `b'`.
/// - if `a < b`, `(a', b')` must be `(0, 1)`.
/// - if `a > b`, `(a', b')` must be `(1, 0)`.
/// - if `a == b`, `(a', b')` could be `(0, 0)` or `(1, 1)`.
///
/// This way of representing the state is nice, because in the case of bits comparison,
/// we can just use `a` and `b` itself without any additional modifications.
///
/// In our algorithm we use pair `(a' == b', a')` because it is easier to recompute,
/// but still doesn't require network communication for initialization.
///
/// When those two bits are known, it is possible to compute the results of all
/// other comparison operations.
///
/// # How is this graph computed?
///
/// - First, we `permuteAxes` so the innermost dimension, which stores bits of
/// numbers become outermost. After that, each component of the array corresponds
/// to one specific bit.
///
/// - Second, if signed numbers are compared, the most significant bit is inverted.
/// It turns out, if we invert that bit, and then compare numbers as usual unsigned
/// numbers, the result will be correct. See the [`flip_msb`] documentation to
/// get more details about this fact.
///
/// - Third, we generate an instance of the `ComparisonResult` for bits.
///
/// - Later, we iteratively shrink `ComparisonResult` to get the result for a whole number
/// instead of the result for each separate bit. On each iteration, we split the result into
/// odd-indexed bits and even-indexed bits, and combine them. If the number of components
/// is not divisible by two, we cut the first component and join it to the existing
/// result.
///
/// # Example
///
/// Let's compare two 5-bit unsigned integers 15 (= 8 + 4 + 2 + 1) and 20 (= 16 + 4).
///
/// | bits     | 0 | 1 | 2 | 3 | 4 |
/// |----------|---|---|---|---|---|
/// |    a     | 1 | 1 | 1 | 1 | 0 |
/// |    b     | 0 | 0 | 1 | 0 | 1 |
/// | a' == b' | 0 | 0 | 1 | 0 | 0 |
///
/// We have 5 components, so 0-th is saved to `res`, 1-st and 2-nd are joined,
/// as well as 3rd and 4th. When components are joined, higher bits have priority.
/// So if we already found that `a != b` based on higher bits, we use that
/// result. Otherwise, use results from smaller bits.
///
/// | components | res | 1..2 | 3..4 |
/// |------------|-----|------|------|
/// |     a'     |  1  |   1  |   0  |
/// |  a' == b'  |  0  |   0  |   0  |
///
/// Number of components is divisible by two, so we only join 1..2, and 3..4,
/// and not change the `res`. We already know the result in the group 3..4,
/// so just copy it.
///
/// | components | res | 1..4 |
/// |------------|-----|------|
/// |     a'     |  1  |   0  |
/// |  a' == b'  |  0  |   0  |
///
/// Only one component left, so join it to `res`.
///
/// | components | res |
/// |------------|-----|
/// |     a'     |  0  |
/// |  a' == b'  |  0  |
///
/// We know that `a' == b'` is `false` and `a'` is `0`. Based on that we can
/// compute the results for other comparison functions.
///
/// For example, if we want to know if `a < b` we can compute `(not a) and (not (a == b))`.
///
#[derive(Clone)]
struct ComparisonResult {
    a_equal_b: Node,
    a: Node,
}

struct ShrinkResult {
    shrinked: Option<ComparisonResult>,
    remainder: Option<ComparisonResult>,
}

impl ComparisonResult {
    fn from_a_b(a: Node, b: Node) -> Result<Self> {
        let graph = a.get_graph();
        let one = constant_scalar(&graph, 1, BIT)?;

        let a_equal_b = a.add(b)?.add(one)?;
        Ok(Self { a_equal_b, a })
    }

    /// `rhs` has higher priority. So if `rhs.a_is_smaller` or `rhs.b_is_smaller`
    /// is set to 1 for a specific position, this value is used. Otherwise, values
    /// from `self` are used.
    ///
    /// Multiplication depth of formulas here is 1, which is important for performance
    /// reasons.
    fn join(&self, rhs: &Self) -> Result<Self> {
        let graph = &self.a_equal_b.get_graph();
        let one = constant_scalar(graph, 1, BIT)?;

        let a = self
            .a
            .multiply(rhs.a_equal_b.clone())?
            .add(rhs.a.multiply(rhs.a_equal_b.add(one)?)?)?;

        let a_equal_b = self.a_equal_b.multiply(rhs.a_equal_b.clone())?;

        Ok(Self { a_equal_b, a })
    }

    /// Joins even-indexed and odd-indexed values
    /// If the number of elements is not divisible by two,
    /// the first element is returned in `ShrinkResult.remainder`.
    fn shrink(&self) -> Result<ShrinkResult> {
        let bit_len = self.a_equal_b.get_type()?.get_shape()[0] as i64;
        let offset = bit_len % 2;
        let remainder = if offset == 0 {
            None
        } else {
            Some(Self {
                a_equal_b: self.a_equal_b.get(vec![0])?,
                a: self.a.get(vec![0])?,
            })
        };
        let shrinked = if bit_len <= 1 {
            None
        } else {
            let slice0 = self.sub_slice(offset, bit_len)?;
            let slice1 = self.sub_slice(offset + 1, bit_len)?;
            Some(slice0.join(&slice1)?)
        };

        Ok(ShrinkResult {
            shrinked,
            remainder,
        })
    }

    /// Returns every second element starting from `start_offset`
    fn sub_slice(&self, start_offset: i64, bit_len: i64) -> Result<Self> {
        // TODO: at some point this could become the slowest part, as getting
        // every second element is not efficient. If we have an efficient way to
        // reorder elements of the array at the beginning, we potentially could
        // do it an a way, such that later all splits will have the form
        // [0..len/2] and [len/2..len].
        // But currently the slowest part is `permuteAxes` and there is no good
        // way of permutating the array, so not optimizing it right now.
        let get_slice = |node: &Node| {
            node.get_slice(vec![SliceElement::SubArray(
                Some(start_offset),
                Some(bit_len),
                Some(2),
            )])
        };
        Ok(Self {
            a_equal_b: get_slice(&self.a_equal_b)?,
            a: get_slice(&self.a)?,
        })
    }

    fn not_a(&self) -> Result<Node> {
        let graph = self.a_equal_b.get_graph();
        graph.custom_op(CustomOperation::new(Not {}), vec![self.a.clone()])
    }

    fn equal(&self) -> Result<Node> {
        Ok(self.a_equal_b.clone())
    }

    fn not_equal(&self) -> Result<Node> {
        let graph = self.a_equal_b.get_graph();
        graph.custom_op(CustomOperation::new(Not {}), vec![self.equal()?])
    }

    fn less_than(&self) -> Result<Node> {
        self.not_a()?.multiply(self.not_equal()?)
    }

    fn greater_than(&self) -> Result<Node> {
        self.a.multiply(self.not_equal()?)
    }

    fn greater_than_equal_to(&self) -> Result<Node> {
        let graph = self.a_equal_b.get_graph();
        graph.custom_op(CustomOperation::new(Not {}), vec![self.less_than()?])
    }

    fn less_than_equal_to(&self) -> Result<Node> {
        let graph = self.a_equal_b.get_graph();
        graph.custom_op(CustomOperation::new(Not {}), vec![self.greater_than()?])
    }
}

/// See [`ComparisonResult`].
///
/// `a` and `b` should have type `Array` with bits pulled out to the outermost dimension.
/// Inputs are interpreted as unsigned numbers. The number of bits should be the same
/// in `a` and `b`.
fn build_comparison_graph(a: Node, b: Node) -> Result<ComparisonResult> {
    let mut to_shrink = ComparisonResult::from_a_b(a, b)?;
    let mut remainders = vec![];
    loop {
        let shrink_res = to_shrink.shrink()?;

        if let Some(remainder) = shrink_res.remainder {
            remainders.push(remainder);
        }

        if let Some(shrinked) = shrink_res.shrinked {
            to_shrink = shrinked;
        } else {
            break;
        }
    }

    let mut res = remainders[0].clone();
    for remainder in remainders[1..].iter() {
        res = res.join(remainder)?;
    }
    Ok(res)
}

fn prepend_dims_with_ones(node: Node, prepend_dims: usize) -> Result<Node> {
    if prepend_dims == 0 {
        return Ok(node);
    }
    let scalar = node.get_type()?.get_scalar_type();
    let mut shape = node.get_type()?.get_shape();
    shape.splice(0..0, vec![1; prepend_dims]);
    node.get_graph().reshape(node, Type::Array(shape, scalar))
}

/// As we support broadcasting in comparison arguments, we need to make
/// sure they are still broadcastable after bits are pulled out to the
/// outermost dimension.
///
/// Say, we want to compare array of size `[2, 3, 64]` and array of size
/// `[3, 64]`. This is a valid operation because `[3, 64]` could be broadcasted
/// to `[2, 3, 64]`.
///
/// After pulling out bits, we get shapes `[64, 2, 3]` and `[64, 3]`, which are not
/// broadcastable anymore.
///
/// To fix this, we convert `[3, 64]` into `[1, 3, 64]` first. After pulling out
/// bits, shape is `[64, 1, 3]`, which could be broadcasted to `[64, 2, 3]`.
fn expand_to_same_dims(a: Node, b: Node) -> Result<(Node, Node)> {
    let len_a = a.get_type()?.get_shape().len();
    let len_b = b.get_type()?.get_shape().len();
    let result_len = max(len_a, len_b);
    let a = prepend_dims_with_ones(a, result_len - len_a)?;
    let b = prepend_dims_with_ones(b, result_len - len_b)?;
    Ok((a, b))
}

/// - This function flips all values of the input Array's last component,
/// which correspond to MSB bit (after `pull_out_bits`), to enable the
/// signed comparisons.
///
/// - Why bit flip is sufficient for obtaining signed comparisons given
/// unsigned comparison functionality? Please see below example:
///
///
/// |sign bit MSB|  b1|  b0| unsigned value|   signed value|
/// |------------|----|----|---------------|---------------|
/// |           0|   0|   0|              0|              0|
/// |           0|   0|   1|              1|              1|
/// |           0|   1|   0|              2|              2|
/// |           0|   1|   1|              3|              3|
/// |           1|   0|   0|              4|             -4|
/// |           1|   0|   1|              5|             -3|
/// |           1|   1|   0|              6|             -2|
/// |           1|   1|   1|              7|             -1|
/// --------------------------------------------------------
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
/// There is also another way of thinking about why bit flip is enough.
///
/// Let `A` be an unsigned integer with the following binary representation
/// `A = | a_(n-1) | ... | a_0 |`.
/// Let `a` be a signed integer with the following binary representations
/// `a = | sign_bit | a_(n-1) | ... | a_0 | = A - sign_bit * 2^n`.
/// Flipping the sign bit and recasting to unsigned results in a shift by `2^n`, i.e.
/// `flib_msb(a) = A + (1 - sign_bit) * 2^n = a + 2^n`.
fn flip_msb(ip: Node) -> Result<Node> {
    let bit_len = ip.get_type()?.get_shape()[0] as i64;

    let magnitude_slice = vec![SliceElement::SubArray(None, Some(-1), Some(1))];
    let graph = ip.get_graph();
    // we want `sign_bit` to be an array, so we can use it in `concatenate` later
    let sign_bit = graph.get_slice(
        ip.clone(),
        vec![SliceElement::SubArray(
            Some(bit_len - 1),
            Some(bit_len),
            None,
        )],
    )?;

    let magnitude_bits = graph.get_slice(ip, magnitude_slice)?;
    let flipped_bit = graph.custom_op(CustomOperation::new(Not {}), vec![sign_bit])?;
    graph.concatenate(vec![magnitude_bits, flipped_bit], 0)
}

/// This function pulls out bits to outermost dimension, and flips MSB for
/// signed comparsions.
///
/// See [`flip_msb`] and [`ComparisonResult`] for details.
fn preprocess_input(signed_comparison: bool, node: Node) -> Result<Node> {
    let node = pull_out_bits(node)?;

    if signed_comparison {
        flip_msb(node)
    } else {
        Ok(node)
    }
}

fn preprocess_inputs(signed_comparison: bool, a: Node, b: Node) -> Result<(Node, Node)> {
    let (a, b) = expand_to_same_dims(a, b)?;
    let a = preprocess_input(signed_comparison, a)?;
    let b = preprocess_input(signed_comparison, b)?;
    Ok((a, b))
}

/// This function validates if the `arguments_types` are suitable for the
/// intended signed custom operation. E.g. there should be at least `2` bits
/// i.e. ( magnitude + sign )
fn validate_signed_arguments(custom_op_name: &str, arguments_types: Vec<Type>) -> Result<()> {
    for (arg_id, arg_type) in arguments_types.iter().enumerate() {
        if *arg_type.get_shape().last().unwrap() < 2 {
            return Err(runtime_error!(
                "{custom_op_name}: Signed input{arg_id} has less than 2 bits"
            ));
        }
    }
    Ok(())
}

/// This function first builds a generic comparison graph, and then
/// calls `post_process_result` to obtain the final result.
/// This functions handles pre-processing of input types to support vectorized inputs.
fn instantiate_comparison_custom_op(
    context: Context,
    arguments_types: Vec<Type>,
    signed_comparison: bool,
    custom_op_name: &str,
    post_process_result: impl FnOnce(&ComparisonResult) -> Result<Node>,
) -> Result<Graph> {
    validate_arguments_in_broadcast_bit_ops(arguments_types.clone(), custom_op_name)?;
    if signed_comparison {
        validate_signed_arguments(custom_op_name, arguments_types.clone())?;
    }

    let graph = context.create_graph()?;
    let a = graph.input(arguments_types[0].clone())?;
    let b = graph.input(arguments_types[1].clone())?;

    let (a, b) = preprocess_inputs(signed_comparison, a, b)?;
    let result = post_process_result(&build_comparison_graph(a, b)?)?;

    graph.set_output_node(result)?;
    graph.finalize()?;
    Ok(graph)
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
        instantiate_comparison_custom_op(
            context,
            arguments_types,
            self.signed_comparison,
            &self.get_name(),
            |res| res.greater_than(),
        )
    }

    fn get_name(&self) -> String {
        format!("GreaterThan(signed_comparison={})", self.signed_comparison)
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
        instantiate_comparison_custom_op(context, arguments_types, false, &self.get_name(), |res| {
            res.not_equal()
        })
    }

    fn get_name(&self) -> String {
        "NotEqual".to_owned()
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
        instantiate_comparison_custom_op(
            context,
            arguments_types,
            self.signed_comparison,
            &self.get_name(),
            |res| res.less_than(),
        )
    }

    fn get_name(&self) -> String {
        format!("LessThan(signed_comparison={})", self.signed_comparison)
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
        instantiate_comparison_custom_op(
            context,
            arguments_types,
            self.signed_comparison,
            &self.get_name(),
            |res| res.less_than_equal_to(),
        )
    }

    fn get_name(&self) -> String {
        format!(
            "LessThanEqualTo(signed_comparison={})",
            self.signed_comparison
        )
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
        instantiate_comparison_custom_op(
            context,
            arguments_types,
            self.signed_comparison,
            &self.get_name(),
            |res| res.greater_than_equal_to(),
        )
    }

    fn get_name(&self) -> String {
        format!(
            "GreaterThanEqualTo(signed_comparison={})",
            self.signed_comparison
        )
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
        instantiate_comparison_custom_op(context, arguments_types, false, &self.get_name(), |res| {
            res.equal()
        })
    }

    fn get_name(&self) -> String {
        "Equal".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::broadcast::broadcast_shapes;
    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::scalar_type;
    use crate::data_types::tuple_type;
    use crate::data_types::ArrayShape;
    use crate::data_types::{
        array_type, ScalarType, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8,
    };
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::inline::inline_common::DepthOptimizationLevel;
    use crate::inline::inline_ops::inline_operations;
    use crate::inline::inline_ops::InlineConfig;
    use crate::inline::inline_ops::InlineMode;

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

    #[test]
    fn test_comparison_graph_size() -> Result<()> {
        let mut custom_ops = vec![];
        custom_ops.push(CustomOperation::new(Equal {}));
        custom_ops.push(CustomOperation::new(NotEqual {}));
        for &signed_comparison in [false, true].iter() {
            custom_ops.push(CustomOperation::new(GreaterThan { signed_comparison }));
            custom_ops.push(CustomOperation::new(LessThan { signed_comparison }));
            custom_ops.push(CustomOperation::new(GreaterThanEqualTo {
                signed_comparison,
            }));
            custom_ops.push(CustomOperation::new(LessThanEqualTo { signed_comparison }));
        }

        for custom_op in custom_ops.into_iter() {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i_a = g.input(array_type(vec![64], BIT))?;
            let i_b = g.input(array_type(vec![64], BIT))?;
            let o = g.custom_op(custom_op, vec![i_a, i_b])?;
            g.set_output_node(o)?;
            g.finalize()?;

            c.set_main_graph(g.clone())?;
            c.finalize()?;

            let inline_config = InlineConfig {
                default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
                ..Default::default()
            };
            let instantiated_context = run_instantiation_pass(c)?.get_context();
            let inlined_context = inline_operations(instantiated_context, inline_config.clone())?;
            let num_nodes = inlined_context.get_main_graph()?.get_num_nodes();

            assert!(num_nodes <= 200);
        }
        Ok(())
    }
}
