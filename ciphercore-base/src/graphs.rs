//! Crucial structs, enums, functions and types to create computation graphs.
use atomic_refcell::AtomicRefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::ptr;
use std::sync::Arc;
use std::sync::Weak;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::constants::type_size_limit_constants;
use crate::custom_ops::CustomOperation;
use crate::data_types::{get_size_estimation_in_bits, ArrayShape, ScalarType, Type};
use crate::data_values::Value;
use crate::errors::Result;
use crate::type_inference::{create_type_inference_worker, TypeInferenceWorker};

use crate::version::{VersionedData, DATA_VERSION};

#[cfg(feature = "py-binding")]
use crate::custom_ops::PyBindingCustomOperation;
#[cfg(feature = "py-binding")]
use crate::data_types::{PyBindingScalarType, PyBindingType};
#[cfg(feature = "py-binding")]
use crate::typed_value::PyBindingTypedValue;
#[cfg(feature = "py-binding")]
use pywrapper_macro::{enum_to_struct_wrapper, fn_wrapper, impl_wrapper, struct_wrapper};

/// This enum represents different types of slice elements that are used to create indexing slices (see [Slice] and [Graph::get_slice]).
///
/// The semantics is similar to [the NumPy slice indexing](https://numpy.org/doc/stable/user/basics.indexing.html).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "py-binding", enum_to_struct_wrapper)]
pub enum SliceElement {
    /// Single index of a given array dimension.
    ///
    /// The index is given by a signed integer. If negative, the index is interpreted as in [NumPy](https://numpy.org/doc/stable/user/basics.indexing.html).
    ///
    /// For example, to choose all the elements of an array with the last index in the first dimension and the first index in the second dimension, one can use a slice `vec![SingleIndex(-1), SingleIndex(0)]`.
    SingleIndex(i64),
    /// Sub-array denotes a range of indices of a given array dimension.
    ///
    /// It follows the description of [the NumPy basic slice](https://numpy.org/doc/stable/user/basics.indexing.html), which is defined by 3 signed integers: `start`, `stop`, `step`. `step` can't be equal to zero.
    ///
    /// For example, to choose all the elements of an array with even indices in the first dimension, one can use a slice `vec![SubArray(Some(0), None, Some(2))].
    SubArray(Option<i64>, Option<i64>, Option<i64>),
    /// Ellipsis denotes several dimensions where indices are not restricted.
    ///
    /// For example, to choose all the elements of an array with index `0` in the first dimension and index `2` in the last dimension, one can use a slice `vec![SingleIndex(0), Ellipsis, SingleIndex(2)]`.
    Ellipsis,
}

/// Slice type denotes an indexing slice (see [NumPy slicing](https://numpy.org/doc/stable/user/basics.indexing.html)).
///
/// It is a vector of slice elements that describes the indices of a sub-array in any appropriate array.
pub type Slice = Vec<SliceElement>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "py-binding", enum_to_struct_wrapper)]
pub enum JoinType {
    Inner,
    Left,
    Union,
    Full,
}

#[doc(hidden)]
#[cfg(feature = "py-binding")]
#[pyo3::pymethods]
impl PyBindingJoinType {
    #[staticmethod]
    pub fn from_inner() -> Self {
        PyBindingJoinType {
            inner: JoinType::Inner,
        }
    }
    #[staticmethod]
    pub fn from_left() -> Self {
        PyBindingJoinType {
            inner: JoinType::Left,
        }
    }
    #[staticmethod]
    pub fn from_union() -> Self {
        PyBindingJoinType {
            inner: JoinType::Union,
        }
    }
}

/// Shard config contains the parameters of the Sharding operation, namely:
///
/// - number of shards into which input dataset will be split,
/// - size of each shard, i.e., the number of rows in each shard,
/// - headers of columns whose rows are hashed to find the index of a shard where the corresponding row will be placed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "py-binding", struct_wrapper)]
pub struct ShardConfig {
    pub num_shards: u64,
    pub shard_size: u64,
    /// headers of columns whose rows are hashed to find the index of a shard where the corresponding row will be placed
    pub shard_headers: Vec<String>,
}

#[doc(hidden)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Operation {
    Input(Type),
    Zeros(Type),
    Ones(Type),
    Add,
    Subtract,
    Multiply,
    // Elementwise multiplication of integer arrays by bit arrays.
    // It leaves an integer array element as is or make it zero it depending on a bit array element.
    MixedMultiply,
    // Dot operation follows the numpy (tensor)dot semantics: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    Dot,
    // Matmul operation follows the numpy matmul semantics: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    // In particular, unlike Dot, it doesn't support scalar inputs.
    Matmul,
    Gemm(bool, bool),
    Truncate(u128),
    Sum(ArrayShape),
    CumSum(u64),
    PermuteAxes(ArrayShape),
    Get(ArrayShape),
    GetSlice(Slice),
    Reshape(Type),
    NOP,
    Random(Type),
    PRF(u64, Type),
    PermutationFromPRF(u64, u64),
    Stack(ArrayShape),
    Concatenate(u64),
    Constant(Type, Value),
    A2B,
    B2A(ScalarType),
    CreateTuple,
    CreateNamedTuple(Vec<String>),
    CreateVector(Type),
    TupleGet(u64),
    NamedTupleGet(String),
    VectorGet,
    Zip,
    Repeat(u64),
    Call,
    Iterate,
    ArrayToVector,
    VectorToArray,
    // Operations that can't be compiled to MPC protocols
    RandomPermutation(u64),
    Gather(u64),
    CuckooHash,
    InversePermutation,
    CuckooToPermutation,
    DecomposeSwitchingMap(u64),
    SegmentCumSum,
    Shard(ShardConfig),
    // SQL joins
    Join(JoinType, HashMap<String, String>),
    ApplyPermutation(bool),
    Sort(String),
    Custom(CustomOperation),
    // Operations used for debugging graphs.
    Print(String),
    Assert(String),
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let operation_name = if let Operation::Custom(custom_op) = self {
            custom_op.get_name()
        } else {
            let operation_w_type_str = format!("{:?}", *self);
            let split_for_operation = operation_w_type_str.split('(');
            let vec_operation_and_types: Vec<&str> = split_for_operation.collect();
            if vec_operation_and_types.is_empty() {
                "-null-".to_owned()
            } else {
                vec_operation_and_types[0].to_owned()
            }
        };
        write!(f, "{operation_name}")
    }
}

impl Operation {
    pub fn is_prf_operation(&self) -> bool {
        matches!(
            self,
            Operation::PRF(_, _) | Operation::PermutationFromPRF(_, _)
        )
    }

    pub fn is_broadcasting_called(&self) -> bool {
        matches!(
            self,
            Operation::Add
                | Operation::Subtract
                | Operation::Multiply
                | Operation::Matmul
                | Operation::Gemm(_, _)
                | Operation::MixedMultiply
                | Operation::Stack(_)
        )
    }

    pub fn is_mpc_compiled(&self) -> bool {
        matches!(
            self,
            Operation::Input(_)
                | Operation::Zeros(_)
                | Operation::Ones(_)
                | Operation::Add
                | Operation::Subtract
                | Operation::Multiply
                | Operation::MixedMultiply
                | Operation::Dot
                | Operation::Matmul
                | Operation::Gemm(_, _)
                | Operation::Truncate(_)
                | Operation::Sum(_)
                | Operation::CumSum(_)
                | Operation::PermuteAxes(_)
                | Operation::Get(_)
                | Operation::GetSlice(_)
                | Operation::Reshape(_)
                | Operation::Stack(_)
                | Operation::Concatenate(_)
                | Operation::Constant(_, _)
                | Operation::A2B
                | Operation::B2A(_)
                | Operation::CreateTuple
                | Operation::CreateNamedTuple(_)
                | Operation::CreateVector(_)
                | Operation::TupleGet(_)
                | Operation::NamedTupleGet(_)
                | Operation::VectorGet
                | Operation::Zip
                | Operation::Repeat(_)
                | Operation::ArrayToVector
                | Operation::VectorToArray
                | Operation::Join(_, _)
                | Operation::ApplyPermutation(_)
                | Operation::Sort(_)
        )
    }

    pub fn update_prf_id(&self, prf_id: u64) -> Result<Self> {
        match self {
            Operation::PRF(_, scalar_type) => Ok(Operation::PRF(prf_id, scalar_type.clone())),
            Operation::PermutationFromPRF(_, size) => {
                Ok(Operation::PermutationFromPRF(prf_id, *size))
            }
            _ => Err(runtime_error!("Operation is not a PRF operation")),
        }
    }

    pub fn is_input(&self) -> bool {
        matches!(self, Operation::Input(_))
    }

    pub fn is_const_optimizable(&self) -> Result<bool> {
        match self {
            // Zeros and Ones exist precisely because we don't want to store them as Constants
            // to keep the graph size small.
            Operation::Zeros(_) | Operation::Ones(_) => Ok(false),
            op => Ok(!op.is_input() && !op.is_randomizing()?),
        }
    }

    // If an operation computes a randomized output, return true
    pub fn is_randomizing(&self) -> Result<bool> {
        match self {
            Operation::Random(_)
            | Operation::RandomPermutation(_)
            | Operation::CuckooToPermutation
            | Operation::DecomposeSwitchingMap(_) => Ok(true),
            Operation::Input(_)
            | Operation::Zeros(_)
            | Operation::Ones(_)
            | Operation::A2B
            | Operation::Add
            | Operation::ApplyPermutation(_)
            | Operation::ArrayToVector
            | Operation::Assert(_)
            | Operation::B2A(_)
            | Operation::Subtract
            | Operation::Multiply
            | Operation::MixedMultiply
            | Operation::Matmul
            | Operation::Dot
            | Operation::Gemm(_, _)
            | Operation::Truncate(_)
            | Operation::Sum(_)
            | Operation::CumSum(_)
            | Operation::Concatenate(_)
            | Operation::CreateNamedTuple(_)
            | Operation::CreateTuple
            | Operation::CreateVector(_)
            | Operation::CuckooHash
            | Operation::SegmentCumSum
            | Operation::PermuteAxes(_)
            | Operation::Get(_)
            | Operation::Gather(_)
            | Operation::GetSlice(_)
            | Operation::Reshape(_)
            | Operation::NOP
            | Operation::InversePermutation
            | Operation::PRF(_, _)
            | Operation::PermutationFromPRF(_, _)
            | Operation::Stack(_)
            | Operation::NamedTupleGet(_)
            | Operation::Sort(_)
            | Operation::TupleGet(_)
            | Operation::Constant(_, _)
            | Operation::VectorGet
            | Operation::Zip
            | Operation::Repeat(_)
            | Operation::VectorToArray
            | Operation::Join(_, _)
            | Operation::Print(_)
            | Operation::Shard(_) => Ok(false),
            Operation::Call | Operation::Iterate => Err(runtime_error!(
                "The status of operations calling other graphs cannot be defined"
            )),
            Operation::Custom(_) => Err(runtime_error!(
                "The status of custom operations cannot be defined"
            )),
        }
    }
}

struct NodeBody {
    graph: WeakGraph,
    node_dependencies: Vec<WeakNode>,
    graph_dependencies: Vec<WeakGraph>,
    operation: Operation,
    id: u64,
}

#[derive(Serialize, Deserialize)]
struct SerializableNodeBody {
    node_dependencies: Vec<u64>,
    graph_dependencies: Vec<u64>,
    operation: Operation,
}

type NodeBodyPointer = Arc<AtomicRefCell<NodeBody>>;

/// A structure that stores a pointer to a computation graph node that corresponds to an operation.
///
/// [Clone] trait duplicates the pointer, not the underlying nodes.
///
/// [PartialEq] trait compares pointers, not the related nodes.
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, BIT};
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = scalar_type(BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t).unwrap();
/// assert!(n1 != n2);
/// let n3 = n1.clone();
/// assert!(n1 == n3);
/// ```
#[cfg_attr(feature = "py-binding", struct_wrapper)]
pub struct Node {
    body: NodeBodyPointer,
}

type SerializableNode = Arc<SerializableNodeBody>;

impl Clone for Node {
    /// Returns a new [Node] value with a copy of the pointer to a node.
    fn clone(&self) -> Self {
        Node {
            body: self.body.clone(),
        }
    }
}

impl PartialEq for Node {
    /// Tests whether `self` and `other` nodes are equal via comparison of their respective pointers.
    ///
    /// # Arguments
    ///
    /// `other` - another [Node] value
    ///
    /// # Returns
    ///
    /// `true` if `self` and `other` are equal, `false` otherwise
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.body, &other.body)
    }
}

impl Eq for Node {}

impl Hash for Node {
    /// Hashes the node pointer.
    ///
    /// # Arguments
    ///
    /// `state` - state of a hash function that is changed after hashing the node
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(&*self.body, state);
    }
}

/// Public methods which supposed to be imported in Python.
#[cfg_attr(feature = "py-binding", impl_wrapper)]
impl Node {
    /// Returns the parent graph that contains the node.
    ///
    /// # Returns
    ///
    /// Parent graph of the node
    pub fn get_graph(&self) -> Graph {
        self.body.borrow().graph.upgrade()
    }

    /// Returns the dependency nodes that are used to compute the value in the current node.
    ///
    /// # Returns
    ///
    /// Vector of nodes used by the node to perform its operation
    pub fn get_node_dependencies(&self) -> Vec<Node> {
        self.body
            .borrow()
            .node_dependencies
            .iter()
            .map(|n| n.upgrade())
            .collect()
    }

    /// Returns the dependency graphs that are used to compute the value in the current node.
    ///
    /// These dependencies are non-empty only for `Call` and `Iterate` operations.
    ///
    /// # Returns
    ///
    /// Vector of graphs used by the node to perform its operation
    pub fn get_graph_dependencies(&self) -> Vec<Graph> {
        self.body
            .borrow()
            .graph_dependencies
            .iter()
            .map(|g| g.upgrade())
            .collect()
    }

    /// Returns the ID of the node.
    ///
    /// A node ID is a serial number of a node between `0` and `n-1` where `n` is the number of nodes in the parent graph.
    /// This number is equal to the number of nodes in the parent graph before this node was added to it.
    ///
    /// # Returns
    ///
    /// Node ID
    pub fn get_id(&self) -> u64 {
        self.body.borrow().id
    }

    /// Returns the pair of the parent graph ID and node ID
    ///
    /// # Returns
    ///
    /// (Graph ID, Node ID)
    pub fn get_global_id(&self) -> (u64, u64) {
        (self.get_graph().get_id(), self.get_id())
    }

    /// Returns the operation associated with the node.
    ///
    /// # Returns
    ///
    /// Operation associated with the node
    pub fn get_operation(&self) -> Operation {
        self.body.borrow().operation.clone()
    }

    /// Returns the type of the value computed by the node.
    ///
    /// # Returns
    ///
    /// Output type of the node operation
    pub fn get_type(&self) -> Result<Type> {
        let context = self.get_graph().get_context();

        {
            let context_body = context.body.borrow();
            if let Some(tc) = &context_body.type_checker {
                if let Some(cached_type) = tc.cached_node_type(self)? {
                    return Ok(cached_type);
                }
            }
        }

        let mut context_body = context.body.borrow_mut();
        if let Some(tc) = &mut context_body.type_checker {
            tc.process_node(self.clone())
        } else {
            Err(runtime_error!("Type checker is not available"))
        }
    }
    /// Applies [Context::set_node_name] to the parent context and `this` node. Returns the clone of `this`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{scalar_type, BIT};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n = g.input(t).unwrap();
    /// n.set_name("XOR").unwrap();
    /// ```
    pub fn set_name(&self, name: &str) -> Result<Node> {
        self.get_graph()
            .get_context()
            .set_node_name(self.clone(), name)?;
        Ok(self.clone())
    }

    /// Applies [Context::get_node_name] to the parent context and `this` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{scalar_type, BIT};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n = g.input(t).unwrap();
    /// n.set_name("XOR").unwrap();
    /// assert_eq!(n.get_name().unwrap(), Some("XOR".to_owned()));
    /// ```
    pub fn get_name(&self) -> Result<Option<String>> {
        self.get_graph().get_context().get_node_name(self.clone())
    }

    /// Adds a node to the parent graph that adds elementwise the array or scalar associated with the node to an array or scalar of the same scalar type associated with another node.
    ///
    /// Applies [Graph::add] to the parent graph, `this` node and the `b` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t).unwrap();
    /// let n3 = n1.add(n2).unwrap();
    /// ```
    pub fn add(&self, b: Node) -> Result<Node> {
        self.get_graph().add(self.clone(), b)
    }

    /// Adds a node to the parent graph that subtracts elementwise the array or scalar of the same scalar type associated with another node from an array or scalar associated with the node.
    ///
    /// Applies [Graph::subtract] to the parent graph, `this` node and the `b` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t).unwrap();
    /// let n3 = n1.subtract(n2).unwrap();
    /// ```
    pub fn subtract(&self, b: Node) -> Result<Node> {
        self.get_graph().subtract(self.clone(), b)
    }

    /// Adds a node to the parent graph that multiplies elementwise the array or scalar associated with the node by an array or scalar of the same scalar type associated with another node.
    ///
    /// Applies [Graph::multiply] to the parent graph, `this` node and the `b` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t).unwrap();
    /// let n3 = n1.multiply(n2).unwrap();
    /// ```
    pub fn multiply(&self, b: Node) -> Result<Node> {
        self.get_graph().multiply(self.clone(), b)
    }

    /// Adds a node to the parent graph that multiplies elementwise the array or scalar associated with the node by a binary array or scalar associated with another node.
    ///
    /// Applies [Graph::mixed_multiply] to the parent graph, `this` node and the `b` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, INT32, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(INT32);
    /// let bit_t = scalar_type(BIT);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.input(bit_t).unwrap();
    /// let n3 = n1.mixed_multiply(n2).unwrap();
    /// ```
    pub fn mixed_multiply(&self, b: Node) -> Result<Node> {
        self.get_graph().mixed_multiply(self.clone(), b)
    }

    /// Adds a node to the parent graph that computes the dot product of arrays or scalars associated with the node and another node.
    ///
    /// Applies [Graph::dot] to the parent graph, `this` node and the `b` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![10], INT32);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t).unwrap();
    /// let n3 = n1.dot(n2).unwrap();
    /// ```
    pub fn dot(&self, b: Node) -> Result<Node> {
        self.get_graph().dot(self.clone(), b)
    }

    /// Adds a node to the parent graph that computes the matrix product of two arrays associated with the node and another node.
    ///
    /// Applies [Graph::matmul] to the parent graph, `this` node and the `b` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![2, 3], INT32);
    /// let t2 = array_type(vec![3, 2], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = n1.matmul(n2).unwrap();
    /// ```
    pub fn matmul(&self, b: Node) -> Result<Node> {
        self.get_graph().matmul(self.clone(), b)
    }

    /// Adds a node to the parent graph that computes the generatl matrix product of two arrays associated with the node and another node.
    ///
    /// Applies [Graph::gemm] to the parent graph, `this` node and the `b` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = n1.gemm(n2, false, true).unwrap();
    /// ```
    #[doc(hidden)]
    pub fn gemm(&self, b: Node, transpose_a: bool, transpose_b: bool) -> Result<Node> {
        self.get_graph()
            .gemm(self.clone(), b, transpose_a, transpose_b)
    }

    /// Adds a node that computes a join of a given type on two named tuples along given key headers.
    /// More detailed documentation can be found in [Graph::join].
    ///
    /// Applies [Graph::join] to the parent graph, `this` node and the `b` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::{create_context, JoinType};
    /// # use ciphercore_base::data_types::{INT32, INT64, UINT8, BIT, array_type, named_tuple_type};
    /// # use ciphercore_base::type_inference::NULL_HEADER;
    /// # use std::collections::HashMap;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1n = array_type(vec![100], BIT);
    /// let t11 = array_type(vec![100], INT32);
    /// let t12 = array_type(vec![100, 128], BIT);
    /// let t13 = array_type(vec![100],  INT64);
    /// let t2n = array_type(vec![50], BIT);
    /// let t21 = array_type(vec![50], INT32);
    /// let t22 = array_type(vec![50, 128], BIT);
    /// let t23 = array_type(vec![50], UINT8);
    /// let t1 = named_tuple_type(vec![
    ///     (NULL_HEADER.to_owned(), t1n),
    ///     ("ID".to_owned(), t11),
    ///     ("Occupation".to_owned(), t12),
    ///     ("Revenue".to_owned(), t13),
    /// ]);
    /// let t2 = named_tuple_type(vec![
    ///     (NULL_HEADER.to_owned(), t2n),
    ///     ("ID".to_owned(), t21),
    ///     ("Job".to_owned(), t22),
    ///     ("Age".to_owned(), t23),
    /// ]);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = n1.join(n2, JoinType::Inner, HashMap::from([
    ///     ("ID".to_owned(), "ID".to_owned()),
    ///     ("Occupation".to_owned(), "Job".to_owned()),
    /// ])).unwrap();
    /// ```
    pub fn join(&self, b: Node, t: JoinType, headers: HashMap<String, String>) -> Result<Node> {
        self.get_graph().join(self.clone(), b, t, headers)
    }

    /// Adds a node that applies a permutation to the array along the first dimension.
    ///
    /// # Arguments
    ///
    /// * `p` - node containing a permutation.
    ///
    /// # Returns
    ///
    /// New permuted node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, UINT64, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![25, 3], INT32);
    /// let a = g.input(t).unwrap();
    /// let p = g.input(array_type(vec![25], UINT64)).unwrap();
    /// let a = a.apply_permutation(p).unwrap();
    /// ```
    #[doc(hidden)]
    pub fn apply_permutation(&self, p: Node) -> Result<Node> {
        self.get_graph().apply_permutation(self.clone(), p)
    }

    /// Adds a node that applies an inverse permutation to the array along the first dimension.
    ///
    /// # Arguments
    ///
    /// * `p` - node containing a permutation.
    ///
    /// # Returns
    ///
    /// New permuted node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, UINT64, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![25, 3], INT32);
    /// let a = g.input(t).unwrap();
    /// let p = g.input(array_type(vec![25], UINT64)).unwrap();
    /// let a = a.apply_inverse_permutation(p).unwrap();
    /// ```
    #[doc(hidden)]
    pub fn apply_inverse_permutation(&self, p: Node) -> Result<Node> {
        self.get_graph().apply_inverse_permutation(self.clone(), p)
    }

    /// Adds a node that sorts a table given as named tuple according to the column given by the key argument.
    /// The key column must be a 2-d BIT array of shape [n, b], interpreted as bitstrings of length b.
    /// Other columns in the named tuple must be arrays of arbitrary type and shape, as long as they
    /// share the first dimension: [n, ...].
    /// Bitstrings are sorted lexicographically, and the sorting algorithm is stable: preserving relative
    /// order of entries in other arrays where the corresponding key entries match.
    ///
    /// # Arguments
    /// * `key` - name of the field to sort on it, this array must be 2-d of type BIT.
    ///
    /// # Returns
    ///
    /// New sorted node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, INT32, UINT64, array_type, named_tuple_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let v1 = g.input(array_type(vec![20], INT32)).unwrap();
    /// let v2 = g.input(array_type(vec![20, 10, 2], UINT64)).unwrap();
    /// let k = g.input(array_type(vec![20, 32], BIT)).unwrap();
    /// let a = g.create_named_tuple(vec![("key".to_string(), k), ("value1".to_string(), v1), ("value2".to_string(), v2)]).unwrap();
    /// let a = a.sort("key".to_string()).unwrap();
    /// ```
    pub fn sort(&self, key: String) -> Result<Node> {
        self.get_graph().sort(self.clone(), key)
    }

    /// Adds a node to the parent graph that divides a scalar or each entry of the array associated with the node by a positive constant integer `scale`.
    ///
    /// Applies [Graph::add] to the parent graph, `this` node and `scale`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.truncate(4).unwrap();
    /// ```
    pub fn truncate(&self, scale: u128) -> Result<Node> {
        self.get_graph().truncate(self.clone(), scale)
    }

    /// Adds a node to the parent graph that computes the sum of entries of the array associated with the node along given axes.
    ///
    /// Applies [Graph::sum] to the parent graph, `this` node and `axes`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let axes = vec![1, 0];
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.sum(axes).unwrap();
    /// ```
    pub fn sum(&self, axes: ArrayShape) -> Result<Node> {
        self.get_graph().sum(self.clone(), axes)
    }

    /// Adds a node to the parent graph that computes the cumulative sum of elements along a given axis.
    ///
    /// Applies [Graph::cum_sum] to the parent graph, `this` node and `axis`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.cum_sum(1).unwrap();
    /// ```
    pub fn cum_sum(&self, axis: u64) -> Result<Node> {
        self.get_graph().cum_sum(self.clone(), axis)
    }

    /// Adds a node to the parent graph that permutes the array associated with the node along given axes.
    ///
    /// Applies [Graph::permute_axes] to the parent graph, `this` node and `axes`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let axes = vec![1, 0, 2];
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.permute_axes(axes).unwrap();
    /// ```
    pub fn permute_axes(&self, axes: ArrayShape) -> Result<Node> {
        self.get_graph().permute_axes(self.clone(), axes)
    }

    /// Adds a node to the parent graph that inverts a given permutation.
    ///
    /// Applies [Graph::inverse_permutation] to the parent graph and `this` node.
    #[doc(hidden)]
    pub fn inverse_permutation(&self) -> Result<Node> {
        self.get_graph().inverse_permutation(self.clone())
    }

    /// Adds a node to the parent graph that extracts a sub-array with a given index from the array associated with the node.
    ///
    /// Applies [Graph::get] to the parent graph, `this` node and `index`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let index = vec![2];
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.get(index).unwrap();
    /// ```
    pub fn get(&self, index: ArrayShape) -> Result<Node> {
        self.get_graph().get(self.clone(), index)
    }

    /// Adds a node that extracts a sub-array corresponding to a given slice from the array associated with the node.
    ///
    /// Applies [Graph::get_slice] to the parent graph, `this` node and `slice`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::{create_context, SliceElement};
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let slice = vec![SliceElement::Ellipsis, SliceElement::SubArray(None, None, Some(-2))];
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.get_slice(slice).unwrap();
    /// ```
    pub fn get_slice(&self, slice: Slice) -> Result<Node> {
        self.get_graph().get_slice(self.clone(), slice)
    }

    /// Adds a node to the parent graph that reshapes a value associated with the node to a given compatible type.
    ///
    /// Applies [Graph::reshape] to the parent graph, `this` node and `new_type`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let old_t = array_type(vec![3, 2, 3], INT32);
    /// let new_t = array_type(vec![3,6], INT32);
    /// let n1 = g.input(old_t).unwrap();
    /// let n2 = n1.reshape(new_t).unwrap();
    /// ```
    pub fn reshape(&self, new_type: Type) -> Result<Node> {
        self.get_graph().reshape(self.clone(), new_type)
    }

    #[doc(hidden)]
    pub fn nop(&self) -> Result<Node> {
        self.get_graph().nop(self.clone())
    }

    #[doc(hidden)]
    pub fn prf(&self, iv: u64, output_type: Type) -> Result<Node> {
        self.get_graph().prf(self.clone(), iv, output_type)
    }

    #[doc(hidden)]
    pub fn permutation_from_prf(&self, iv: u64, n: u64) -> Result<Node> {
        self.get_graph().permutation_from_prf(self.clone(), iv, n)
    }

    /// Adds a node to the parent graph converting an integer array or scalar associated with the node to the binary form.
    ///
    /// Applies [Graph::a2b] to the parent graph and `this` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.a2b().unwrap();
    /// ```
    pub fn a2b(&self) -> Result<Node> {
        self.get_graph().a2b(self.clone())
    }

    /// Adds a node to the parent graph converting a binary array associated with the node to an array of a given scalar type.
    ///
    /// Applies [Graph::b2a] to the parent graph, `this` node and `scalar_type`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 32], BIT);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.b2a(INT32).unwrap();
    /// ```
    pub fn b2a(&self, scalar_type: ScalarType) -> Result<Node> {
        self.get_graph().b2a(self.clone(), scalar_type)
    }

    /// Adds a node that extracts an element of a tuple associated with the node.
    ///
    /// Applies [Graph::tuple_get] to the parent graph, `this` node and `index`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![3, 2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.create_tuple(vec![n1, n2]).unwrap();
    /// let n4 = n3.tuple_get(1).unwrap();
    /// ```
    pub fn tuple_get(&self, index: u64) -> Result<Node> {
        self.get_graph().tuple_get(self.clone(), index)
    }

    /// Adds a node to the parent graph that extracts an element of a named tuple associated with the node.
    ///
    /// Applies [Graph::named_tuple_get] to the parent graph, `this` node and the `key` string.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![3, 2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.create_named_tuple(vec![("node1".to_owned(), n1), ("node2".to_owned(), n2)]).unwrap();
    /// let n4 = n3.named_tuple_get("node2".to_owned()).unwrap();
    /// ```
    pub fn named_tuple_get(&self, key: String) -> Result<Node> {
        self.get_graph().named_tuple_get(self.clone(), key)
    }

    /// Adds a node to the parent graph that extracts an element of a vector associated with the node.
    ///
    /// Applies [Graph::vector_get] to the parent graph, `this` node and the `index` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{UINT32, INT32, array_type, scalar_type};
    /// # use ciphercore_base::data_values::Value;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t.clone()).unwrap();
    /// let n3 = g.create_vector(t, vec![n1,n2]).unwrap();
    /// let index = g.constant(scalar_type(UINT32), Value::from_scalar(0, UINT32).unwrap()).unwrap();
    /// let n4 = n3.vector_get(index).unwrap();
    /// ```
    pub fn vector_get(&self, index: Node) -> Result<Node> {
        self.get_graph().vector_get(self.clone(), index)
    }

    /// Adds a node to the parent graph converting an array associated with the node to a vector.
    ///
    /// Applies [Graph::array_to_vector] to the parent graph and `this` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, scalar_type, INT32, UINT32};
    /// # use ciphercore_base::data_values::Value;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![4, 3, 2], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.array_to_vector(n1).unwrap();
    /// let index = g.constant(scalar_type(UINT32), Value::from_scalar(0, UINT32).unwrap()).unwrap();
    /// let n3 = n2.vector_get(index).unwrap();
    ///
    /// assert!(n2.get_type().unwrap().is_vector());
    /// assert_eq!(n3.get_type().unwrap().get_shape(), vec![3,2]);
    /// ```
    pub fn array_to_vector(&self) -> Result<Node> {
        self.get_graph().array_to_vector(self.clone())
    }

    /// Adds a node to the parent graph converting a vector associated with the node to an array.
    ///
    /// Applies [Graph::vector_to_array] to the parent graph and `this` node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, vector_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let vec_t = vector_type(4, t);
    /// let n1 = g.input(vec_t).unwrap();
    /// let n2 = n1.vector_to_array().unwrap();
    ///
    /// assert!(n2.get_type().unwrap().is_array());
    /// assert_eq!(n2.get_type().unwrap().get_shape(), vec![4, 3, 2]);
    /// ```
    pub fn vector_to_array(&self) -> Result<Node> {
        self.get_graph().vector_to_array(self.clone())
    }

    /// Adds a node to the parent graph converting a vector associated with the node to an array.
    ///
    /// Applies [Graph::gather] to the parent graph and `this` node.
    pub fn gather(&self, indices: Node, axis: u64) -> Result<Node> {
        self.get_graph().gather(self.clone(), indices, axis)
    }

    /// Adds a node that creates a vector with `n` copies of a value of this node.
    ///
    /// Applies [Graph::repeat] to the parent graph, `this` node and `n`.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = n1.repeat(10).unwrap();
    /// ```
    pub fn repeat(&self, n: u64) -> Result<Node> {
        self.get_graph().repeat(self.clone(), n)
    }

    /// Adds a node returning the Cuckoo hash map of an input array of binary strings using provided hash functions.
    ///
    /// Applies [Graph::cuckoo_hash] to the parent graph, `this` node and `hash_matrices`.
    #[doc(hidden)]
    pub fn cuckoo_hash(&self, hash_matrices: Node) -> Result<Node> {
        self.get_graph().cuckoo_hash(self.clone(), hash_matrices)
    }

    /// Adds a node that, given an input multidimensional array A, binary one-dimensional array B (first dimension is n in both array) and starting value v, computes the following iteration
    ///
    /// output[i] = A[i-1] + B[i-1] * output[i-1]
    ///
    /// where i in {1,...,n} and output[0] = v.
    ///
    /// Applies [Graph::segment_cumsum] to the parent graph, `this` node, `binary_array` and `first_row`.
    #[doc(hidden)]
    pub fn segment_cumsum(&self, binary_array: Node, first_row: Node) -> Result<Node> {
        self.get_graph()
            .segment_cumsum(self.clone(), binary_array, first_row)
    }

    /// Adds a node that computes sharding of a given table according to a given sharding config.
    /// Sharding config contains names of the columns whose hashed values are used for sharding.
    ///
    /// Each shard is accompanied by a Boolean mask indicating whether a corresponding row stems from the input table or padded (1 if a row comes from input).
    ///
    /// Applies [Graph::shard] to the parent graph, `this` node and `shard_config`.
    #[doc(hidden)]
    pub fn shard(&self, shard_config: ShardConfig) -> Result<Node> {
        self.get_graph().shard(self.clone(), shard_config)
    }

    /// Adds a node that converts a switching map array into a tuple of the following components:
    /// - a permutation map array with deletion,
    /// - a duplication map array,
    /// - a permutation map array without deletion.
    ///
    /// The composition of these maps is equal to the input switching map, which is an array containing non-unique indices of some array.
    ///
    /// Applies [Graph::decompose_switching_map] to the parent graph and `this`.
    #[doc(hidden)]
    pub fn decompose_switching_map(&self, n: u64) -> Result<Node> {
        self.get_graph().decompose_switching_map(self.clone(), n)
    }

    /// Adds a node that converts a Cuckoo hash table to a random permutation.
    ///
    /// Applies [Graph::cuckoo_to_permutation] to the parent graph and `this` node.
    #[doc(hidden)]
    pub fn cuckoo_to_permutation(&self) -> Result<Node> {
        self.get_graph().cuckoo_to_permutation(self.clone())
    }

    /// Adds an operation which logs the value of the node at runtime.
    pub fn print(&self, message: String) -> Result<Node> {
        self.get_graph().print(message, self.clone())
    }

    /// Applies [Graph::set_output_node] to the parent graph and `this` node.
    ///
    /// # Returns
    ///
    /// This node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, vector_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let vec_t = vector_type(4, t);
    /// let n1 = g.input(vec_t).unwrap();
    /// let n2 = g.vector_to_array(n1).unwrap();
    /// n2.set_as_output().unwrap();
    /// g.finalize().unwrap();
    /// ```
    pub fn set_as_output(&self) -> Result<Node> {
        self.get_graph().set_output_node(self.clone())?;
        Ok(self.clone())
    }
}

/// Methods which aren't supposed to be imported in Python.
impl Node {
    fn make_serializable(&self) -> SerializableNode {
        Arc::new(SerializableNodeBody {
            node_dependencies: self
                .get_node_dependencies()
                .iter()
                .map(|n| n.get_id())
                .collect(),
            graph_dependencies: self
                .get_graph_dependencies()
                .iter()
                .map(|n| n.get_id())
                .collect(),
            operation: self.get_operation(),
        })
    }

    fn downgrade(&self) -> WeakNode {
        WeakNode {
            body: Arc::downgrade(&self.body),
        }
    }

    #[doc(hidden)]
    pub fn add_annotation(&self, annotation: NodeAnnotation) -> Result<Node> {
        self.get_graph()
            .get_context()
            .add_node_annotation(self, annotation)?;
        Ok(self.clone())
    }

    #[doc(hidden)]
    pub fn get_annotations(&self) -> Result<Vec<NodeAnnotation>> {
        self.get_graph()
            .get_context()
            .get_node_annotations(self.clone())
    }
}
type WeakNodeBodyPointer = Weak<AtomicRefCell<NodeBody>>;

struct WeakNode {
    body: WeakNodeBodyPointer,
}

impl WeakNode {
    //upgrade function panics if the the Node pointer it downgraded from went out of scope
    fn upgrade(&self) -> Node {
        Node {
            body: self.body.upgrade().unwrap(),
        }
    }
}

impl Clone for WeakNode {
    fn clone(&self) -> Self {
        WeakNode {
            body: self.body.clone(),
        }
    }
}

struct GraphBody {
    finalized: bool,
    nodes: Vec<Node>,
    output_node: Option<WeakNode>,
    id: u64,
    context: WeakContext,
}

#[derive(Serialize, Deserialize)]
struct SerializableGraphBody {
    finalized: bool,
    nodes: Vec<SerializableNode>,
    output_node: Option<u64>,
}

type GraphBodyPointer = Arc<AtomicRefCell<GraphBody>>;

/// A structure that stores a pointer to a computation graph, where every node corresponds to an operation.
///
/// # Rust crates
///
/// [Clone] trait duplicates the pointer, not the underlying graph.
///
/// [PartialEq] trait compares pointers, not the related graphs.
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// let c = create_context().unwrap();
/// let g1 = c.create_graph().unwrap();
/// let g2 = c.create_graph().unwrap();
/// assert_ne!(g1, g2);
/// let g3 = g1.clone();
/// assert_eq!(g1, g3);
/// ```
#[cfg_attr(feature = "py-binding", struct_wrapper)]
pub struct Graph {
    body: GraphBodyPointer,
}

type SerializableGraph = Arc<SerializableGraphBody>;

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("body", &self.body.as_ptr())
            .finish()
    }
}

impl Clone for Graph {
    /// Returns a new [Graph] value with a copy of the pointer to a computation graph.
    fn clone(&self) -> Self {
        Graph {
            body: self.body.clone(),
        }
    }
}

impl PartialEq for Graph {
    /// Tests whether `self` and `other` graphs are equal via comparison of their respective pointers.
    ///
    /// # Arguments
    ///
    /// `other` - another [Graph] value
    ///
    /// # Returns
    ///
    /// `true` if `self` and `other` are equal, `false` otherwise
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.body, &other.body)
    }
}

impl Eq for Graph {}

impl Hash for Graph {
    /// Hashes the graph pointer.
    ///
    /// # Arguments
    ///
    /// `state` - state of a hash function that is changed after hashing the graph
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(&*self.body, state);
    }
}

/// Public methods which supposed to be imported in Python.
#[cfg_attr(feature = "py-binding", impl_wrapper)]
impl Graph {
    /// Applies [Context::set_main_graph] to the parent context and `this` graph. Returns the clone of `this`.
    ///
    /// # Returns
    ///
    /// This graph
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let n = g.input(t).unwrap();
    /// n.set_as_output().unwrap();
    /// g.finalize().unwrap();
    /// g.set_as_main().unwrap();
    /// ```
    pub fn set_as_main(&self) -> Result<Graph> {
        self.get_context().set_main_graph(self.clone())?;
        Ok(self.clone())
    }

    /// Applies [Context::set_graph_name] to the parent context and `this` graph. Returns the clone of `this`.
    ///
    /// # Arguments
    ///
    /// `name` - name of the graph
    ///
    /// # Returns
    ///
    /// This graph
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// g.set_name("relu").unwrap();
    /// ```
    pub fn set_name(&self, name: &str) -> Result<Graph> {
        self.get_context().set_graph_name(self.clone(), name)?;
        Ok(self.clone())
    }

    /// Applies [Context::get_graph_name] to the parent context and `this` graph.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// g.set_name("relu").unwrap();
    /// assert_eq!(g.get_name().unwrap(), "relu".to_owned());
    /// ```
    pub fn get_name(&self) -> Result<String> {
        self.get_context().get_graph_name(self.clone())
    }

    /// Applies [Context::retrieve_node] to the parent context and `this` graph.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let n = g.input(scalar_type(BIT)).unwrap();
    /// n.set_name("input_node").unwrap();
    /// assert!(n == g.retrieve_node("input_node").unwrap());
    /// ```
    pub fn retrieve_node(&self, name: &str) -> Result<Node> {
        self.get_context().retrieve_node(self.clone(), name)
    }

    /// Adds an input node to the graph and returns it.
    ///
    /// During evaluation, input nodes require values to be supplied.
    ///
    /// # Arguments
    ///
    /// `input_type` - type of a new input node
    ///
    /// # Returns
    ///
    /// New input node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n = g.input(t).unwrap();
    /// ```
    pub fn input(&self, input_type: Type) -> Result<Node> {
        self.add_node(vec![], vec![], Operation::Input(input_type))
    }

    /// Adds an node with zeros of given type.
    ///
    /// Compared to `constant` this node does produce a big value array in serialized graph.
    ///
    /// # Arguments
    ///
    /// `t` - node type
    ///
    /// # Returns
    ///
    /// New node with zeros of given type.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{UINT8, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let z = g.zeros(array_type(vec![10, 20], UINT8)).unwrap();
    /// ```
    pub fn zeros(&self, t: Type) -> Result<Node> {
        self.add_node(vec![], vec![], Operation::Zeros(t))
    }

    /// Adds an node with ones of given type.
    ///
    /// Compared to `constant` this node does produce a big value array in serialized graph.
    ///
    /// # Arguments
    ///
    /// `t` - node type
    ///
    /// # Returns
    ///
    /// New node with ones of given type.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{UINT8, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let z = g.ones(array_type(vec![10, 20], UINT8)).unwrap();
    /// ```
    pub fn ones(&self, t: Type) -> Result<Node> {
        self.add_node(vec![], vec![], Operation::Ones(t))
    }

    /// Adds a node that sums two arrays or scalars of the same scalar type elementwise.
    ///
    /// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)). For example, adding two arrays of shapes `[10,1,7]` and `[8,1]` results in an array of shape `[10,8,7]`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing the first term (array or scalar)
    /// * `b` - node containing the second term (array or scalar)
    ///
    /// # Returns
    ///
    /// New addition node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t).unwrap();
    /// let n3 = g.add(n1, n2).unwrap();
    /// ```
    pub fn add(&self, a: Node, b: Node) -> Result<Node> {
        self.add_node(vec![a, b], vec![], Operation::Add)
    }

    /// Adds a node that subtracts two arrays or scalars of the same scalar type elementwise.
    ///
    /// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)). For example, subtracting two arrays of shapes `[10,1,7]` and `[8,1]` results in an array of shape `[10,8,7]`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing the minuend (array or scalar)
    /// * `b` - node containing the subtrahend (array or scalar)
    ///
    /// # Returns
    ///
    /// New subtraction node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t).unwrap();
    /// let n3 = g.subtract(n1, n2).unwrap();
    /// ```
    pub fn subtract(&self, a: Node, b: Node) -> Result<Node> {
        self.add_node(vec![a, b], vec![], Operation::Subtract)
    }

    /// Adds a node that multiplies two arrays or scalars of the same scalar type elementwise.
    ///
    /// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)). For example, multiplication of two arrays of shapes `[10,1,7]` and `[8,1]` results in an array of shape `[10,8,7]`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing the first factor (array or scalar)
    /// * `b` - node containing the second factor (array or scalar)
    ///
    /// # Returns
    ///
    /// New multiplication node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t).unwrap();
    /// let n3 = g.multiply(n1, n2).unwrap();
    /// ```
    pub fn multiply(&self, a: Node, b: Node) -> Result<Node> {
        self.add_node(vec![a, b], vec![], Operation::Multiply)
    }

    /// Adds a node that multiplies an integer array or scalar by a binary array or scalar elementwise.
    /// For each integer element, this operation returns this element or zero depending on the corresponding bit element.
    ///
    /// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)). For example, multiplication of two arrays of shapes `[10,1,7]` and `[8,1]` results in an array of shape `[10,8,7]`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an integer array or scalar
    /// * `b` - node containing a binary array or scalar
    ///
    /// # Returns
    ///
    /// New mixed multiplication node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, INT32, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = scalar_type(INT32);
    /// let t2 = scalar_type(BIT);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.mixed_multiply(n1, n2).unwrap();
    /// ```
    pub fn mixed_multiply(&self, a: Node, b: Node) -> Result<Node> {
        self.add_node(vec![a, b], vec![], Operation::MixedMultiply)
    }

    /// Adds a node that computes the dot product according to [the NumPy rules](https://numpy.org/doc/stable/reference/generated/numpy.dot.html):
    /// * if both factors are 1-dimensional arrays, return their inner product;
    /// * if both factors are 2-dimensional arrays, return their matrix product;
    /// * if one of the factors is scalar, return the result of [multiply](Graph::multiply);
    /// * if the first factor is n-dimensional and the second one is 1-dimensional,
    /// compute the elementwise multiplication and return the sum over the last axis.
    /// * if both factors are n-dimensional (n>2), return the sum product
    /// over the last axis of the first factor and the second-to-last axis of the second factor, i.e.
    ///
    /// `dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])` (in the NumPy notation).
    ///
    /// # Arguments
    ///
    /// * `a` - node containing the first factor (array or scalar)
    /// * `b` - node containing the second factor (array or scalar)
    ///
    /// # Returns
    ///
    /// New dot product node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![10], INT32);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t).unwrap();
    /// let n3 = g.dot(n1, n2).unwrap();
    /// ```
    pub fn dot(&self, a: Node, b: Node) -> Result<Node> {
        self.add_node(vec![a, b], vec![], Operation::Dot)
    }

    /// Adds a node that computes the matrix product of two arrays according to [the NumPy rules](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
    ///
    /// Each array is represented as an array of 2-dimensional matrix elements and this node returns the elementwise product of such matrix arrays.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing the first array
    /// * `b` - node containing the second array
    ///
    /// # Returns
    ///
    /// New matrix product node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![2, 3], INT32);
    /// let t2 = array_type(vec![3, 2], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.matmul(n1, n2).unwrap();
    /// ```
    pub fn matmul(&self, a: Node, b: Node) -> Result<Node> {
        self.add_node(vec![a, b], vec![], Operation::Matmul)
    }

    /// Adds a node that computes the general matrix product of two arrays according to [the ONNX rules](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm) with `alpha = 1`, `beta = 0` and `C = 0`.
    ///
    /// Each array is represented as an array of 2-dimensional matrix elements and this node returns the elementwise product of such matrix arrays.
    /// Each matrix should have at least 2 dimensions.
    /// To multiply by 1-dimensional matrices (i.e., vectors), please resort to `matmul` or `dot`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing the first array
    /// * `b` - node containing the second array
    /// * `transpose_a` - if true, the first array will be transposed
    /// * `transpose_b` - if true, the second array will be transposed
    ///
    /// # Returns
    ///
    /// New Gemm node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.gemm(n1, n2, false, true).unwrap();
    /// ```
    #[doc(hidden)]
    pub fn gemm(&self, a: Node, b: Node, transpose_a: bool, transpose_b: bool) -> Result<Node> {
        self.add_node(
            vec![a, b],
            vec![],
            Operation::Gemm(transpose_a, transpose_b),
        )
    }

    /// Adds a node that computes a join of a given type on two named tuples along given key headers.
    ///
    /// Each tuple should consist of arrays having the same number of rows, i.e. the first dimensions of these arrays should be equal.
    /// **WARNING**: The rows consisiting of only columns with given key headers (key columns) should be unique.
    ///  
    /// In addition, each named tuple should have a binary array named with NULL_HEADER that contains zeros in rows void of content; otherwise, it contains ones.
    /// This column is called the null column.
    ///
    /// This operation returns:
    /// - Inner join: a named tuple containing rows whose content is equal in the key columns named by given key headers.
    /// - Left join: a named tuple containing rows of the first named tuple and the rows of the second named tuple whose content is equal to the one of the first named tuple in the key columns named by given key headers.
    /// - Union join: a named tuple containing rows of the first named tuple that are not in the inner join and all the rows of the second set.
    /// In contrast to the SQL union, this operation does not require that input datasets have respective columns of the same type.
    /// This means that columns of both datasets are included and filled with zeros where no data can be retrieved.
    /// Namely, the rows of the second set in the union join will contain zeros in non-key columns of the first set and vice versa.
    /// - Full join: a named tuple containing all the rows of the both sets.
    /// If a row of the first set match with a row of the second set, they are merged into one.
    /// The order of rows goes as follows:
    /// 1. the rows of the first set that don't belong to the inner join.
    /// 2. all the rows of the second set including those merged with the rows of the first set as in inner join.
    /// In this form, full join is computed as `union_join(a, left_join(b, a))`.  
    ///
    /// The content of non-key columns is merged.
    /// The order of these rows is the same as in the first named tuple.
    /// The content of other rows is set to zero including the null column.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing the first named tuple
    /// * `b` - node containing the second named tuple
    /// * `t` - join type (Inner/Left/Union/Full)
    /// * `headers` - headers of columns along which the join is performed
    ///
    /// # Returns
    ///
    /// New join node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::{create_context, JoinType};
    /// # use ciphercore_base::data_types::{INT32, INT64, UINT8, BIT, array_type, named_tuple_type};
    /// # use ciphercore_base::type_inference::NULL_HEADER;
    /// # use std::collections::HashMap;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1n = array_type(vec![100], BIT);
    /// let t11 = array_type(vec![100], INT32);
    /// let t12 = array_type(vec![100, 128], BIT);
    /// let t13 = array_type(vec![100],  INT64);
    /// let t2n = array_type(vec![50], BIT);
    /// let t21 = array_type(vec![50], INT32);
    /// let t22 = array_type(vec![50, 128], BIT);
    /// let t23 = array_type(vec![50], UINT8);
    /// let t1 = named_tuple_type(vec![
    ///     (NULL_HEADER.to_owned(), t1n),
    ///     ("ID".to_owned(), t11),
    ///     ("Occupation".to_owned(), t12),
    ///     ("Revenue".to_owned(), t13),
    /// ]);
    /// let t2 = named_tuple_type(vec![
    ///     (NULL_HEADER.to_owned(), t2n),
    ///     ("ID".to_owned(), t21),
    ///     ("Job".to_owned(), t22),
    ///     ("Age".to_owned(), t23),
    /// ]);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.join(n1, n2, JoinType::Inner, HashMap::from([
    ///     ("ID".to_owned(), "ID".to_owned()),
    ///     ("Occupation".to_owned(), "Job".to_owned()),
    /// ])).unwrap();
    /// ```
    pub fn join(
        &self,
        a: Node,
        b: Node,
        t: JoinType,
        headers: HashMap<String, String>,
    ) -> Result<Node> {
        self.add_node(vec![a, b], vec![], Operation::Join(t, headers))
    }

    /// Adds a node that applies a permutation to the array along the first dimension.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an array to permute.
    /// * `p` - node containing a permutation.
    ///
    /// # Returns
    ///
    /// New permuted node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, UINT64, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![25, 3], INT32);
    /// let a = g.input(t).unwrap();
    /// let p = g.input(array_type(vec![25], UINT64)).unwrap();
    /// let a = g.apply_permutation(a, p).unwrap();
    /// ```
    #[doc(hidden)]
    pub fn apply_permutation(&self, a: Node, p: Node) -> Result<Node> {
        self.add_node(vec![a, p], vec![], Operation::ApplyPermutation(false))
    }

    /// Adds a node that applies an inverse permutation to the array along the first dimension.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an array to permute.
    /// * `p` - node containing a permutation.
    ///
    /// # Returns
    ///
    /// New permuted node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, UINT64, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![25, 3], INT32);
    /// let a = g.input(t).unwrap();
    /// let p = g.input(array_type(vec![25], UINT64)).unwrap();
    /// let a = g.apply_inverse_permutation(a, p).unwrap();
    /// ```
    #[doc(hidden)]
    pub fn apply_inverse_permutation(&self, a: Node, p: Node) -> Result<Node> {
        self.add_node(vec![a, p], vec![], Operation::ApplyPermutation(true))
    }

    /// Adds a node that sorts a table given as named tuple according to the column given by the key argument.
    /// The key column must be a 2-d BIT array of shape [n, b], interpreted as bitstrings of length b.
    /// Other columns in the named tuple must be arrays of arbitrary type and shape, as long as they
    /// share the first dimension: [n, ...].
    /// Bitstrings are sorted lexicographically, and the sorting algorithm is stable: preserving relative
    /// order of entries in other arrays where the corresponding key entries match.
    ///
    /// # Arguments
    /// * `a` - node containing a named tuple -- arrays to sort.
    /// * `key` - name of the field to sort on it, this array must be 2-d of type BIT.
    ///
    /// # Returns
    ///
    /// New sorted node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, INT32, UINT64, array_type, named_tuple_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let v1 = g.input(array_type(vec![20], INT32)).unwrap();
    /// let v2 = g.input(array_type(vec![20, 10, 2], UINT64)).unwrap();
    /// let k = g.input(array_type(vec![20, 32], BIT)).unwrap();
    /// let a = g.create_named_tuple(vec![("key".to_string(), k), ("value1".to_string(), v1), ("value2".to_string(), v2)]).unwrap();
    /// let a = g.sort(a, "key".to_string()).unwrap();
    /// ```
    pub fn sort(&self, a: Node, key: String) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::Sort(key))
    }

    /// Adds a node that divides a scalar or each entry of an array by a positive constant integer `scale`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing a scalar or an array
    /// * `scale` - positive integer
    ///
    /// # Returns
    ///
    /// New truncate node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.truncate(n1, 4).unwrap();
    /// ```
    pub fn truncate(&self, a: Node, scale: u128) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::Truncate(scale))
    }

    /// Adds a node that computes the sum of entries of an array along given axes (see [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)).
    ///
    /// For example, summing the array `[[1000, 200], [30, 4]]` along the first or the second axes results in the arrays `[1030,204]` or `[1200,34]`, respectively. Summing along both axes yields `1234`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an array
    /// * `axes` - indices of the axes of `a`
    ///
    /// # Returns
    ///
    /// New sum node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let axes = vec![1, 0];
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.sum(n1, axes).unwrap();
    /// ```
    pub fn sum(&self, a: Node, axes: ArrayShape) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::Sum(axes))
    }

    /// Adds a node that computes the cumulative sum of elements along a given axis. (see [numpy.cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)).
    ///
    /// For example, summing the array `[[1000, 200], [30, 4]]` along the first or the second axes results in the arrays `[[1000, 200], [1030, 204]]` or `[[1000, 1200], [30, 34]]`, respectively.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an array
    /// * `axis` - axis along which the cumulative sum is computed
    ///
    /// # Returns
    ///
    /// New cumulative sum node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.cum_sum(n1, 1).unwrap();
    /// ```
    pub fn cum_sum(&self, a: Node, axis: u64) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::CumSum(axis))
    }

    /// Adds a node that permutes an array along given axes (see [numpy.transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)). This function generalizes matrix transposition.
    ///
    /// For example, permutation of an array of shape `[a,b,c]` with permutation `[2,0,1]` results in an array of shape `[c,a,b]`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an array
    /// * `axes` - indices of the axes of `a`
    ///
    /// # Returns
    ///
    /// New node with permuted axes
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let axes = vec![1, 0, 2];
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.permute_axes(n1, axes).unwrap();
    /// ```
    pub fn permute_axes(&self, a: Node, axes: ArrayShape) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::PermuteAxes(axes))
    }

    /// Adds a node to the parent graph that inverts a given permutation.
    ///
    /// An input permutation should be given by a 1-dimensional array of length n, containing unique integers between 0 and n-1.
    /// The i-th element of an output array is output[i] = j if input[j] = i.
    ///
    /// This operation could be realized through [the Scatter operation](https://en.wikipedia.org/wiki/Gather-scatter_(vector_addressing)#Scatter).
    /// However, the Scatter operation poses a security risk as the corresponding map should hide empty output positions.
    /// This is usually done by padding an input array with dummy values such that its size is equal to the output size.
    /// Then, the Scatter map can be turned into a permutation, which can be easily split into a composition of random permutation maps.
    /// But permutation maps can be performed by Gather, thus making Scatter unnecessary.
    ///
    /// **WARNING**: this function should not be used before MPC compilation.
    ///
    /// # Arguments
    ///
    /// `a` - node containing an array with permutation.
    #[doc(hidden)]
    pub fn inverse_permutation(&self, a: Node) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::InversePermutation)
    }

    /// Adds a node that extracts a sub-array with a given index. This is a special case of [get_slice](Graph::get_slice) and corresponds to single element indexing as in [NumPy](https://numpy.org/doc/stable/user/basics.indexing.html).
    ///
    /// For example, given an array `A` of shape `[a,b,c,d]`, its subarray `B` of shape `[c,d]` with index `[i,j]` can be extracted as follows
    ///
    /// `B = A[i,j,:,:]` (in the NumPy notation)
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an array
    /// * `index` - index of a sub-array
    ///
    /// # Returns
    ///
    /// New node containing an extracted sub-array
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let index = vec![2];
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.get(n1, index).unwrap();
    /// ```
    pub fn get(&self, a: Node, index: ArrayShape) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::Get(index))
    }

    /// Adds a node that extracts a sub-array corresponding to a given slice.
    ///
    /// Our slicing conventions follow [the NumPy rules](https://numpy.org/doc/stable/user/basics.indexing.html).
    ///
    /// For example, given an array `A` of shape `[a,b]`, its subarray `B` containing only the last 3 rows of `A` can be extracted as follows
    ///
    /// `get_slice(A, [-3::])[i,j] = A[a-3+i,j]`.
    ///
    /// Slices are defined as vectors of [SliceElements](SliceElement) that have 3 possible types:
    ///
    /// * [SingleIndex(`i64`)](SliceElement::SingleIndex) is used to extract all the elements with a given index in a respective dimension,
    /// * [SubArray(`Option<i64>, Option<i64>, Option<i64>`)](SliceElement::SubArray) describes the range of indices that should be extracted over a certain dimension (similar to the `a:b:c` notation in [NumPy](https://numpy.org/doc/stable/user/basics.indexing.html))
    /// * [Ellipsis](SliceElement::Ellipsis) describes several consecutive dimensions that must be extracted in full, e.g. the slice `[i,...,j]` can be used to extract all the elements with the index `i` in the first dimension and the index `j` in the last one, while the indices of all the other dimensions have no constraints. See [the NumPy slicing](https://numpy.org/doc/stable/user/basics.indexing.html) for more details.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an array
    /// * `slice` - array slice
    ///
    /// # Returns
    ///
    /// New node containing an extracted sub-array
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::{create_context, SliceElement};
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let slice = vec![SliceElement::Ellipsis, SliceElement::SubArray(None, None, Some(-2))];
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.get_slice(n1, slice).unwrap();
    /// ```
    pub fn get_slice(&self, a: Node, slice: Slice) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::GetSlice(slice))
    }

    /// Adds a node that reshapes a value to a given compatible type (similar to [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html?highlight=reshape#numpy.ndarray.reshape), but more general). Specifically,
    ///
    /// * if the input value is an array, it can be reshaped to any array with the same number of elements;
    /// * if the input value in the flattened form contains `n` arrays or scalars, it can be reshaped to any type with the same number of arrays and scalars. Each array can be reshaped as in the above rule.
    ///
    /// For example, an array of shape `[3,10,5]` can be reshaped to `[2,75]`. A tuple with arrays of shapes `[3,4]`, `[12]`, `[2,6]` can be reshaped to a vector with 3 array elements of shape `[2,2,3]`.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing a value
    /// * `new_type` - type
    ///
    /// # Returns
    ///
    /// New node with a reshaped value
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let old_t = array_type(vec![3, 2, 3], INT32);
    /// let new_t = array_type(vec![3,6], INT32);
    /// let n1 = g.input(old_t).unwrap();
    /// let n2 = g.reshape(n1, new_t).unwrap();
    /// ```
    pub fn reshape(&self, a: Node, new_type: Type) -> Result<Node> {
        let size_estimate = get_size_estimation_in_bits(new_type.clone());
        if size_estimate.is_err() {
            return Err(runtime_error!(
                "Trying to add a reshape node with invalid type size: {:?}",
                size_estimate
            ));
        }
        if size_estimate? > type_size_limit_constants::MAX_INDIVIDUAL_NODE_SIZE {
            return Err(runtime_error!(
                "Trying to add a reshape node larger than MAX_INDIVIDUAL_NODE_SIZE"
            ));
        }
        self.add_node(vec![a], vec![], Operation::Reshape(new_type))
    }

    /// Adds a node creating a random value of a given type.
    ///
    /// **WARNING**: this function should not be used before MPC compilation.
    ///
    /// # Arguments
    ///
    /// `output_type` - type of a constant
    ///
    /// # Returns
    ///
    /// New random node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n = g.random(t).unwrap();
    /// ```
    #[doc(hidden)]
    pub fn random(&self, output_type: Type) -> Result<Node> {
        self.add_node(vec![], vec![], Operation::Random(output_type))
    }

    /// Adds a node creating a random permutation map of a one-dimensional array of length `n`.
    ///
    /// This operation generates a random array of all 64-bit integers from 0 to n-1 in random order.
    ///
    /// **WARNING**: this function should not be used before MPC compilation.
    ///
    /// # Arguments
    ///
    /// `n` - length of permutation
    ///
    /// # Returns
    ///
    /// New random permutation node
    #[doc(hidden)]
    pub fn random_permutation(&self, n: u64) -> Result<Node> {
        self.add_node(vec![], vec![], Operation::RandomPermutation(n))
    }

    /// Adds a node returning the Cuckoo hash map of an input array of binary strings using provided hash functions.
    ///
    /// Hash functions are defined as an array of binary matrices.
    /// The hash of an input string is a product of one of these matrices and this string.
    /// Hence, the last dimension of these matrices should coincide with the length of input strings.
    ///
    /// Random matrices yield a better success probability of hashing.
    ///
    /// If the input array has shape `[..., n, b]` and hash matrices are given as an `[h, m, b]`-array,
    /// then the hash map is an array of shape `[..., 2^m]`.
    /// The hash table element with index `[..., i]` is equal to `j` if the `[..., j]`-th input `b`-bit string is hashed to `i` by some of the given hash functions.
    ///
    /// The number of hash matrices (the first dimension of hash matrices) must be at least 3.
    ///
    /// A bigger ratio `m/n` leads to higher success probability (recommended one is `>=2`)    
    ///
    /// **WARNING**: this function should not be used before MPC compilation.
    ///
    /// # Arguments
    ///
    /// - `array` - input array of binary strings of shape [..., n, b]
    /// - `hash_matrices` - random binary [h, m, b]-array.
    ///
    /// # Returns
    ///
    /// New CuckooHash node
    #[doc(hidden)]
    pub fn cuckoo_hash(&self, array: Node, hash_matrices: Node) -> Result<Node> {
        self.add_node(vec![array, hash_matrices], vec![], Operation::CuckooHash)
    }

    /// Adds a node that, given an input multidimensional array A, binary one-dimensional array B (first dimension is n in both array) and starting value v, computes the following iteration
    ///
    /// output[i] = A[i-1] + B[i-1] * output[i-1]
    ///
    /// where i in {1,...,n} and output[0] = v.
    /// This is similar to computing cumulative sums of consecutive elements (segments) of the input array A.
    /// The locations of these segments are defined by the binary array B.
    ///
    /// This iteration is used in the Duplication protocol (see mpc::mpc_psi) and is done locally by one of the computing parties.
    ///
    /// **WARNING**: this function should not be used before MPC compilation.
    ///
    /// # Arguments
    ///
    /// - `input_array` - input array whose rows are summed within the iteration
    /// - `binary_array` - binary array indicating whether a row of the input array should be added to a previous row of the output array
    /// - `first_row` - first row of the output array
    ///
    /// # Returns
    ///
    /// New SegmentCumSum node containing the output array
    #[doc(hidden)]
    pub fn segment_cumsum(
        &self,
        input_array: Node,
        binary_array: Node,
        first_row: Node,
    ) -> Result<Node> {
        self.add_node(
            vec![input_array, binary_array, first_row],
            vec![],
            Operation::SegmentCumSum,
        )
    }

    /// Adds a node that computes sharding of a given table according to a given sharding config.
    /// Sharding config contains names of the columns whose hashed values are used for sharding.
    /// The size of each shard (i.e., the number of rows) and the number of shards is given in the sharding config.
    /// The number of shards should be smaller than 700.
    ///
    ///
    /// If some resulting shards don't have `shard_size` elements, they're padded with zeros to reach this size.
    /// If the size of some shards exceeds `shard_size`, sharding fails.
    ///
    /// To choose these parameters, consult [the following paper](http://wwwmayr.informatik.tu-muenchen.de/personen/raab/publ/balls.pdf).
    /// Note that for large shard sizes and small number of shards, it holds that
    ///
    /// `shard_size = num_input_rows / num_shards + alpha * sqrt(2 * num_input_rows / num_shards * log(num_shards))`.
    ///
    /// With `alpha = 2`, it is possible to achieve failure probability 2^(-40) if `num_shards < 700` and `shard_size > 2^17`.
    ///
    ///
    /// Each shard is accompanied by a Boolean mask indicating whether a corresponding row stems from the input table or padded (1 if a row comes from input).
    /// The output is given in the form of a tuple of `(mask, shard)`, where `mask` is a binary array and `shard` is a table, i.e., named tuple.
    ///
    /// **WARNING**: this function cannot be compiled to an MPC protocol.
    ///
    /// # Arguments
    ///
    /// - `input_table` - named tuple of arrays containing data for sharding
    /// - `shard_config` - parameters of sharding: number of shards, shard size and names of columns that are hashed in sharding
    ///
    /// # Returns
    ///
    /// New Shard node containing a tuple of shards
    #[doc(hidden)]
    pub fn shard(&self, input_table: Node, shard_config: ShardConfig) -> Result<Node> {
        self.add_node(vec![input_table], vec![], Operation::Shard(shard_config))
    }

    /// Adds a node that converts a switching map array into a random tuple of the following components:
    /// - a permutation map array with deletion (some indices of this map are uniformly random, see below),
    /// - a tuple of duplication map array and duplication bits,
    /// - a permutation map array without deletion.
    ///
    /// The composition of these maps is equal to the input switching map, which is an array containing non-unique indices of some array.
    ///
    /// To create a permutation with deletion, this operation first groups identical indices of the input map together and shifts other indices accordingly, e.g.
    ///
    /// [1, 4, 5, 7, 2, 4] -> [1, 4, 4, 5, 7, 2].
    ///
    /// This can be done by permutation p = [1, 2, 6, 3, 4, 5].
    /// Then, it replaces copies with unique random indices not present in the switching map, e.g.
    ///
    /// [1, 4, 4, 5, 7, 2] -> [1, 4, 3, 5, 7, 2].
    ///
    ///
    /// A duplication map is a tuple of two one-dimensional arrays of length `n`.
    /// The first array contains indices from `[0,n]` in the increasing order with possible repetitions.
    /// The second array contains only zeros and ones.
    /// If its i-th element is zero, it means that the duplication map doesn't change the i-th element of an array it acts upon.
    /// If map's i-th element is one, then the map copies the previous element of the result.
    /// This rules can be summarized by the following equation
    ///
    /// duplication_indices[i] = duplication_bits[i] * duplication_indices[i-1] + (1 - duplication_bits[i]) * i.
    ///
    /// A duplication map is created from the above switching map with grouped indices, replacing the first index occurrence with 0 and other copies with 1, e.g.
    ///
    ///  [1, 4, 4, 5, 7, 2] -> ([0, 1, 1, 3, 4, 5], [0, 0, 1, 0, 0, 0]).
    ///
    /// The last permutation is the inverse of the above permutation p, i.e.
    ///
    /// [1, 2, 4, 5, 6, 3].
    ///
    /// This operation supports vectorization.
    ///
    /// **WARNING**: this function should not be used before MPC compilation.
    ///
    /// # Arguments
    ///
    /// - `switching_map` - an array of one-dimensional arrays containing non-unique indices of some array of length `n` (usually a simple hash table),
    /// - `n` - length of an array that can be mapped by the above switching map.
    ///
    /// # Returns
    ///
    /// New DecomposeSwitchingMap node
    #[doc(hidden)]
    pub fn decompose_switching_map(&self, switching_map: Node, n: u64) -> Result<Node> {
        self.add_node(
            vec![switching_map],
            vec![],
            Operation::DecomposeSwitchingMap(n),
        )
    }

    /// Adds a node that converts a Cuckoo hash table to a random permutation.
    ///
    /// Conversion is done via replacing dummy hash elements by random indices such that the resulting array constitute a permutation.
    ///
    /// **WARNING**: this function should not be used before MPC compilation.
    ///
    /// # Arguments
    ///
    /// `cuckoo_map` - an array containing a Cuckoo hash map with dummy values
    ///
    /// # Returns
    ///
    /// New CuckooToPermutation node
    #[doc(hidden)]
    pub fn cuckoo_to_permutation(&self, cuckoo_map: Node) -> Result<Node> {
        self.add_node(vec![cuckoo_map], vec![], Operation::CuckooToPermutation)
    }

    /// Adds a node that joins a sequence of arrays governed by a given shape.
    ///
    /// The input arrays should have the same shape or be able to be broadcast to the same shape.
    ///
    /// For example, stacking 2 arrays of shapes `[2,2]` and `[2,1]` with the outer shape `[2]` works as follows
    ///
    /// `stack(arrays=[[[1,2],[3,4]], [5,6]], shape=[2]) = [[[1,2],[3,4]], [[5,5], [6,6]]]`
    ///
    /// # Arguments
    ///
    /// * `nodes` - vector of nodes containing arrays
    /// * `outer_shape` - shape defining how the input arrays are arranged in the resulting array
    ///
    /// # Returns
    ///
    /// New stack node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![3, 2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let shape = vec![2];
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.stack(vec![n1,n2], shape).unwrap();
    /// ```
    pub fn stack(&self, nodes: Vec<Node>, outer_shape: ArrayShape) -> Result<Node> {
        self.add_node(nodes, vec![], Operation::Stack(outer_shape))
    }

    /// Adds a node that joins a sequence of arrays along a given axis.
    /// This operation is similar to [the NumPy concatenate](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html).
    ///
    /// The input arrays should have the same shape except in the given axis.
    ///
    /// # Arguments
    ///
    /// * `nodes` - vector of nodes containing arrays
    /// * `axis` - axis along which the above arrays are joined
    ///
    /// # Returns
    ///
    /// New Concatenate node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![3, 2, 3], INT32);
    /// let t2 = array_type(vec![3, 2, 10], INT32);
    /// let shape = vec![2];
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.concatenate(vec![n1,n2], 2).unwrap();
    /// ```
    pub fn concatenate(&self, nodes: Vec<Node>, axis: u64) -> Result<Node> {
        self.add_node(nodes, vec![], Operation::Concatenate(axis))
    }

    /// Adds a node creating a constant of a given type and value.
    ///
    /// # Arguments
    ///
    /// * `output_type` - type of a constant
    /// * `value` - value of a constant
    ///
    /// # Returns
    ///
    /// New constant node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// # use ciphercore_base::data_values::Value;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let v = Value::from_scalar(0, BIT).unwrap();
    /// let n = g.constant(t, v).unwrap();
    /// ```
    pub fn constant(&self, output_type: Type, value: Value) -> Result<Node> {
        self.add_node(vec![], vec![], Operation::Constant(output_type, value))
    }

    /// Adds a node converting an integer array or scalar to the binary form.
    ///
    /// Given an array of shape `[a,b,c]` and scalar type `st`, this node returns an array of shape `[a,b,c,s]` where `s` is the bit size of `st`. For example, an array of shape `[1,2,3]` with `INT32` entries will be converted to a binary array of shape `[1,2,3,32]`.
    ///
    /// # Arguments
    ///
    /// `a` - node containing an array or scalar
    ///
    /// # Returns
    ///
    /// New node converting an array/scalar to the binary form
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.a2b(n1).unwrap();
    /// ```
    pub fn a2b(&self, a: Node) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::A2B)
    }

    /// Adds a node converting a binary array to an array of a given scalar type.
    ///
    /// Given a binary array of shape `[a,b,c]` and a scalar type `st` of bit size `c`, this node returns an array of shape `[a,b]` with `st` entries. For example, a binary array of shape `[2,3,32]` can be converted to an array of shape `[2,3]` with `INT32` entries.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing an array or scalar
    /// * `scalar_type` - scalar type
    ///
    /// # Returns
    ///
    /// New node converting an array from the binary form
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 32], BIT);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.b2a(n1, INT32).unwrap();
    /// ```
    pub fn b2a(&self, a: Node, scalar_type: ScalarType) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::B2A(scalar_type))
    }

    /// Adds a node that creates a tuple from several (possibly, zero) elements.
    ///
    /// # Arguments
    ///
    /// `elements` - vector of nodes
    ///
    /// # Returns
    ///
    /// New node with a tuple
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![3, 2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.create_tuple(vec![n1,n2]).unwrap();
    /// ```
    pub fn create_tuple(&self, elements: Vec<Node>) -> Result<Node> {
        self.add_node(elements, vec![], Operation::CreateTuple)
    }

    /// Adds a node that creates a vector from several (possibly, zero) elements of the same type.
    ///
    /// # Arguments
    ///
    /// `elements` - vector of nodes
    ///
    /// # Returns
    ///
    /// New node with a created vector
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t.clone()).unwrap();
    /// let n3 = g.create_vector(t, vec![n1,n2]).unwrap();
    /// ```
    pub fn create_vector(&self, element_type: Type, elements: Vec<Node>) -> Result<Node> {
        self.add_node(elements, vec![], Operation::CreateVector(element_type))
    }

    /// Adds a node that creates a named tuple from several (possibly, zero) elements.
    ///
    /// # Arguments
    ///
    /// `elements` - vector of pairs (node name, node)
    ///
    /// # Returns
    ///
    /// New node creating a named tuple
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![3, 2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.create_named_tuple(vec![("node1".to_owned(), n1), ("node2".to_owned(), n2)]).unwrap();
    /// ```
    pub fn create_named_tuple(&self, elements: Vec<(String, Node)>) -> Result<Node> {
        let mut nodes = vec![];
        let mut names = vec![];
        for (name, node) in elements {
            nodes.push(node);
            names.push(name);
        }
        self.add_node(nodes, vec![], Operation::CreateNamedTuple(names))
    }

    /// Adds a node that extracts an element of a tuple.
    ///
    /// # Arguments
    ///
    /// * `tuple` - node containing a tuple
    /// * `index` - index of a tuple element between 0 and tuple length minus 1
    ///
    /// # Returns
    ///
    /// New node with an extracted element
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![3, 2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.create_tuple(vec![n1, n2]).unwrap();
    /// let n4 = g.tuple_get(n3, 1).unwrap();
    /// ```
    pub fn tuple_get(&self, tuple: Node, index: u64) -> Result<Node> {
        self.add_node(vec![tuple], vec![], Operation::TupleGet(index))
    }

    /// Adds a node that extracts an element of a named tuple.
    ///
    /// # Arguments
    ///
    /// * `tuple` - node containing a named tuple
    /// * `key` - key of a tuple element
    ///
    /// # Returns
    ///
    /// New node extracting a tuple element
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t1 = array_type(vec![3, 2, 3], INT32);
    /// let t2 = array_type(vec![2, 3], INT32);
    /// let n1 = g.input(t1).unwrap();
    /// let n2 = g.input(t2).unwrap();
    /// let n3 = g.create_named_tuple(vec![("node1".to_owned(), n1), ("node2".to_owned(), n2)]).unwrap();
    /// let n4 = g.named_tuple_get(n3, "node2".to_owned()).unwrap();
    /// ```
    pub fn named_tuple_get(&self, tuple: Node, key: String) -> Result<Node> {
        self.add_node(vec![tuple], vec![], Operation::NamedTupleGet(key))
    }

    /// Adds a node that extracts an element of a vector.
    ///
    /// # Arguments
    ///
    /// * `vec` - node containing a vector
    /// * `index` - node containing the index of a tuple element
    ///
    /// # Returns
    ///
    /// New node extracting a vector element
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{UINT32, INT32, array_type, scalar_type};
    /// # use ciphercore_base::data_values::Value;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let n1 = g.input(t.clone()).unwrap();
    /// let n2 = g.input(t.clone()).unwrap();
    /// let n3 = g.create_vector(t, vec![n1,n2]).unwrap();
    /// let index = g.constant(scalar_type(UINT32), Value::from_scalar(0, UINT32).unwrap()).unwrap();
    /// let n4 = g.vector_get(n3, index).unwrap();
    /// ```
    pub fn vector_get(&self, vec: Node, index: Node) -> Result<Node> {
        self.add_node(vec![vec, index], vec![], Operation::VectorGet)
    }

    /// Adds a node that takes vectors V<sub>1</sub>(n, t<sub>1</sub>), V<sub>2</sub>(n, t<sub>2</sub>), ..., V<sub>k</sub>(n, t<sub>k</sub>) of the same length and returns a vector V(n, tuple(t<sub>1</sub>, ..., t<sub>k</sub>)) (similar to [zip](https://doc.rust-lang.org/stable/std/iter/fn.zip.html)).
    ///
    /// # Arguments
    ///
    /// `nodes` - vector of nodes containing input vectors
    ///
    /// # Returns
    ///
    /// New zip node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, array_type, vector_type};
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let vec_t = vector_type(3, t);
    /// let n1 = g.input(vec_t.clone()).unwrap();
    /// let n2 = g.input(vec_t.clone()).unwrap();
    /// let n3 = g.zip(vec![n1,n2]).unwrap();
    /// ```
    pub fn zip(&self, nodes: Vec<Node>) -> Result<Node> {
        self.add_node(nodes, vec![], Operation::Zip)
    }

    /// Adds a node that creates a vector with `n` copies of a value of a given node.
    ///
    /// # Arguments
    ///
    /// * `a` - node containing a value
    /// * `n` - number of copies
    ///
    /// # Returns
    ///
    /// New repeat node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.repeat(n1, 10).unwrap();
    /// ```
    pub fn repeat(&self, a: Node, n: u64) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::Repeat(n))
    }

    /// Adds a node that calls another graph with inputs contained in given nodes.
    ///
    /// The input graph must be finalized and have as many inputs as the number of provided arguments.
    ///
    /// For example, let `G` be a graph implementing the function `max(x,0)`, then `call(G, [17]) = max(17, 0)`.
    ///
    /// # Arguments
    ///
    /// * `graph` - graph with `n` input nodes
    /// * `arguments` - vector of `n` nodes
    ///
    /// # Returns
    ///
    /// New call node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    ///
    /// let g1 = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2, 3], INT32);
    /// let n1 = g1.input(t.clone()).unwrap();
    /// let n2 = g1.repeat(n1, 10).unwrap();
    /// let n3 = g1.vector_to_array(n2).unwrap();
    /// n3.set_as_output().unwrap();
    /// g1.finalize().unwrap();
    ///
    /// let g2 = c.create_graph().unwrap();
    /// let n4 = g2.input(t).unwrap();
    /// let n5 = g2.add(n4.clone(), n4).unwrap();
    /// let n6 = g2.call(g1, vec![n5]).unwrap();
    /// ```
    pub fn call(&self, graph: Graph, arguments: Vec<Node>) -> Result<Node> {
        self.add_node(arguments, vec![graph], Operation::Call)
    }

    /// Adds a node that iteratively computes a given finalized graph on the elements of a given vector and updates the state value accordingly.
    ///
    /// This node calls another `graph` with 2 input nodes `old_state` and `input` and an output node that returns a [tuple](Type::Tuple) `(new_state, output)`. This graph is used to map the elements of a given vector `V` to another vector `W` as follows:
    /// ```text
    /// graph(state_0, V[0]) -> (state1, W[0]),
    /// graph(state_1, V[1]) -> (state2, W[1]),
    /// ...
    /// graph(state_k, V[k]) -> (final_state, W[k]).
    /// ```
    /// The output is a [tuple](Type::Tuple) `(final_state, W)`. The initial state `state_0` should be provided as an argument.
    ///
    /// This node generalize `map` and `reduce` procedures (see [MapReduce](https://en.wikipedia.org/wiki/MapReduce) for more details).
    ///
    /// For example, let `G` be a graph implementing the function `max(x,0)` and incrementing `state` if its output is negative, then `iterate(G, 0, [-1,2,0,3,2]) = (1, [0,2,0,3,2])`. The final state is equal to the number of negative values in the input vector.
    ///
    /// # Arguments
    ///
    /// * `graph` - graph with 2 input nodes of types T<sub>s</sub> and T<sub>i</sub> and returning a tuple of type (T<sub>s</sub>, T<sub>o</sub>)
    /// * `state` - node containing an initial state of type T<sub>s</sub>
    /// * `input` - node containing a vector with elements of type T<sub>i</sub>
    ///
    /// # Returns
    ///
    /// New iterate node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, BIT, scalar_type, vector_type};
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::ops::utils::constant_scalar;
    /// let c = create_context().unwrap();
    ///
    /// let t_s = scalar_type(BIT);
    /// let t = scalar_type(INT32);
    /// let vec_t = vector_type(10, t.clone());
    ///
    /// // Graph that outputs 0 at even indices or input value at odd indices.
    /// let g1 = c.create_graph().unwrap();
    /// {
    ///     let old_state = g1.input(t_s.clone()).unwrap();
    ///     let input = g1.input(t.clone()).unwrap();
    ///     let result = g1.mixed_multiply(input, old_state.clone()).unwrap();
    ///     let new_state = g1.add(old_state, constant_scalar(&g1, 1, BIT).unwrap()).unwrap();
    ///     let out_tuple = g1.create_tuple(vec![new_state, result]).unwrap();
    ///     out_tuple.set_as_output().unwrap();
    ///     g1.finalize().unwrap();
    /// }
    ///
    /// let g2 = c.create_graph().unwrap();
    /// let initial_state = constant_scalar(&g2, 0, BIT).unwrap();
    /// let input_vector = g2.input(vec_t).unwrap();
    /// g2.iterate(g1, initial_state, input_vector).unwrap();
    /// ```
    pub fn iterate(&self, graph: Graph, state: Node, input: Node) -> Result<Node> {
        self.add_node(vec![state, input], vec![graph], Operation::Iterate)
    }

    /// Adds a node converting an array to a vector.
    ///
    /// Given an array of shape `[a,b,c]`, this node returns a vector of `a` arrays of shape `[b,c]`.
    ///
    /// # Arguments
    ///
    /// `a` - node containing an array
    ///
    /// # Returns
    ///
    /// New node converting an array to a vector
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, scalar_type, INT32, UINT32};
    /// # use ciphercore_base::data_values::Value;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![4, 3, 2], INT32);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.array_to_vector(n1).unwrap();
    /// let index = g.constant(scalar_type(UINT32), Value::from_scalar(0, UINT32).unwrap()).unwrap();
    /// let n3 = g.vector_get(n2.clone(), index).unwrap();
    ///
    /// assert!(n2.get_type().unwrap().is_vector());
    /// assert_eq!(n3.get_type().unwrap().get_shape(), vec![3,2]);
    /// ```    
    pub fn array_to_vector(&self, a: Node) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::ArrayToVector)
    }

    /// Adds a node converting a vector to an array.
    ///
    /// Given a vector of `a` arrays of shape `[b,c]`, this node returns an array of shape `[a,b,c]`.
    ///
    /// # Arguments
    ///
    /// `a` - node containing a vector
    ///
    /// # Returns
    ///
    /// New node converting a vector to an array
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, vector_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let vec_t = vector_type(4, t);
    /// let n1 = g.input(vec_t).unwrap();
    /// let n2 = g.vector_to_array(n1).unwrap();
    ///
    /// assert!(n2.get_type().unwrap().is_array());
    /// assert_eq!(n2.get_type().unwrap().get_shape(), vec![4, 3, 2]);
    /// ```
    pub fn vector_to_array(&self, a: Node) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::VectorToArray)
    }

    /// Adds a node creating an array from the elements of an input array indexed by another array along a given axis.
    ///
    /// Given an input array, this node replaces the dimension `axis` with the dimensions introduced by the indexing array.
    ///
    /// Indices must be unique to prevent possible duplication of shares/ciphertexts.
    /// Such duplicates might cause devastating data leakage.
    ///
    /// This operation is similar to [the NumPy take operation](https://numpy.org/doc/stable/reference/generated/numpy.take.html).
    ///
    /// **WARNING**: this function should not be used before MPC compilation.
    ///
    /// # Arguments
    ///
    /// `input` - node containing an input array
    /// `indices` - node containing indices
    /// `axis` - index of the axis along which indices are chosen
    ///
    /// # Returns
    ///
    /// New Gather node
    #[doc(hidden)]
    pub fn gather(&self, input: Node, indices: Node, axis: u64) -> Result<Node> {
        self.add_node(vec![input, indices], vec![], Operation::Gather(axis))
    }

    /// Checks that the graph has an output node and finalizes the graph.
    ///
    /// After finalization the graph can't be changed.
    ///
    /// # Returns
    ///
    /// Finalized graph
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, vector_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let vec_t = vector_type(4, t);
    /// let n1 = g.input(vec_t).unwrap();
    /// let n2 = g.vector_to_array(n1).unwrap();
    /// n2.set_as_output().unwrap();
    /// g.finalize().unwrap();
    /// ```
    pub fn finalize(&self) -> Result<Graph> {
        let output_node = self.body.borrow_mut().output_node.clone();
        match output_node {
            Some(_) => {
                self.body.borrow_mut().finalized = true;
                Ok(self.clone())
            }
            None => Err(runtime_error!("Output node is not set")),
        }
    }

    /// Returns the vector of nodes contained in the graph in order of construction.
    ///
    /// # Returns
    ///
    /// Vector of nodes of the graph
    pub fn get_nodes(&self) -> Vec<Node> {
        self.body.borrow().nodes.clone()
    }

    /// Promotes a given node to the output node of the parent graph.
    ///
    /// # Arguments
    ///
    /// `output_node` - node to be set as output
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, vector_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let vec_t = vector_type(4, t);
    /// let n1 = g.input(vec_t).unwrap();
    /// let n2 = g.vector_to_array(n1).unwrap();
    /// g.set_output_node(n2).unwrap();
    /// g.finalize().unwrap();
    /// ```
    pub fn set_output_node(&self, output_node: Node) -> Result<()> {
        let current_output_node = self.body.borrow().output_node.clone();
        match current_output_node {
            Some(_) => Err(runtime_error!("Output node is already set")),
            None => {
                if output_node.get_graph() != *self {
                    Err(runtime_error!("Output node has to be from the same graph"))
                } else {
                    self.body.borrow_mut().output_node = Some(output_node.downgrade());
                    Ok(())
                }
            }
        }
    }

    /// Returns the output node of the graph.
    ///
    /// # Returns
    ///
    /// Output node of the graph
    pub fn get_output_node(&self) -> Result<Node> {
        let current_output_node = self.body.borrow().output_node.clone();
        match current_output_node {
            Some(output_node) => Ok(output_node.upgrade()),
            None => Err(runtime_error!("Output node is not set")),
        }
    }

    /// Returns the ID of the graph.
    ///
    /// A graph ID is a serial number of a graph between `0` and `n-1` where `n` is the number of graphs in the parent context.
    ///
    /// # Returns
    ///
    /// Graph ID
    pub fn get_id(&self) -> u64 {
        self.body.borrow().id
    }

    /// Returns the number of the graph nodes.
    ///
    /// # Returns
    ///
    /// Number of the graph nodes
    pub fn get_num_nodes(&self) -> u64 {
        self.body.borrow().nodes.len() as u64
    }

    /// Returns the node corresponding to a given ID.
    ///
    /// # Arguments
    ///
    /// `id` - node ID
    ///
    /// # Returns
    ///
    /// Node with a given ID
    pub fn get_node_by_id(&self, id: u64) -> Result<Node> {
        let nodes = &self.body.borrow().nodes;
        if id >= nodes.len() as u64 {
            Err(runtime_error!("Invalid id for the node retrieval"))
        } else {
            Ok(nodes[id as usize].clone())
        }
    }

    /// Returns the context of the graph nodes.
    ///
    /// # Returns
    ///
    /// Context of the graph
    pub fn get_context(&self) -> Context {
        self.body.borrow().context.upgrade()
    }

    /// Adds a node computing a given custom operation.
    ///
    /// Custom operations can be created by the user as public structs implementing the [CustomOperationBody](../custom_ops/trait.CustomOperationBody.html).
    ///
    /// # Arguments
    ///
    /// * `op` - custom operation
    /// * `arguments` - vector of nodes used as input for the custom operation
    ///
    /// # Returns
    ///
    /// New custom operation node
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, BIT};
    /// # use ciphercore_base::custom_ops::{CustomOperation, Not};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], BIT);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.custom_op(CustomOperation::new(Not {}), vec![n1]).unwrap();
    /// ```
    pub fn custom_op(&self, op: CustomOperation, arguments: Vec<Node>) -> Result<Node> {
        self.add_node(arguments, vec![], Operation::Custom(op))
    }

    /// Adds a node which logs its input at runtime, and returns the input.
    /// This is intended to be used for debugging.
    ///
    /// # Arguments
    ///
    /// * `message` - Informational message to be printed
    /// * `input` - Node to be printed
    ///
    /// # Returns
    ///
    /// The value of the node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, BIT};
    /// # use ciphercore_base::custom_ops::{CustomOperation, Not};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], BIT);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.print("n1:".into(), n1).unwrap();
    /// ```
    pub fn print(&self, message: String, input: Node) -> Result<Node> {
        self.add_node(vec![input], vec![], Operation::Print(message))
    }

    /// Adds a node which fails the execution at runtime if `condition` is false, and returns the `input` otherwise.
    /// This is intended to be used for debugging.
    ///
    /// # Arguments
    ///
    /// * `message` - message to be returned for the failed assertion.
    /// * `condition` - BIT to be checked in the assertion.
    /// * `input` - Node to be returned for pass-through.
    ///
    /// # Returns
    ///
    /// The value of the node.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, scalar_type, BIT};
    /// # use ciphercore_base::custom_ops::{CustomOperation, Not};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let cond = g.input(scalar_type(BIT)).unwrap();
    /// let t = array_type(vec![3, 2], BIT);
    /// let n1 = g.input(t).unwrap();
    /// let n2 = g.assert("Condition".into(), cond, n1).unwrap();
    /// ```
    pub fn assert(&self, message: String, condition: Node, input: Node) -> Result<Node> {
        self.add_node(vec![condition, input], vec![], Operation::Assert(message))
    }
}

/// Methods which aren't supposed to be imported in Python.
impl Graph {
    /// Adds an operation node to the graph and returns it.
    ///
    /// # Arguments
    ///
    /// * `node_dependencies` - vector of nodes necessary to perform the given operation
    /// * `graph_dependencies` - vector of graphs necessary to perform the given operation
    /// * `operation` - operation performed by the node
    ///
    /// # Returns
    ///
    /// New operation node that gets added
    pub fn add_node(
        &self,
        node_dependencies: Vec<Node>,
        graph_dependencies: Vec<Graph>,
        operation: Operation,
    ) -> Result<Node> {
        if self.is_finalized() {
            return Err(runtime_error!("Can't add a node to a finalized graph"));
        }
        for dependency in &node_dependencies {
            if dependency.get_graph() != *self
                || dependency.get_id() >= self.body.borrow().nodes.len() as u64
                || self.body.borrow().nodes[dependency.get_id() as usize] != *dependency
            {
                return Err(runtime_error!(
                    "Can't add a node with invalid node dependencies"
                ));
            }
        }
        for dependency in &graph_dependencies {
            if !dependency.is_finalized() {
                return Err(runtime_error!(
                    "Can't add a node with not finilized graph dependency"
                ));
            }
            if dependency.get_id() >= self.get_id() {
                return Err(runtime_error!(
                    "Can't add a node with graph dependency with bigger id. {} >= {}",
                    dependency.get_id(),
                    self.get_id()
                ));
            }
            if dependency.get_context() != self.get_context() {
                return Err(runtime_error!(
                    "Can't add a node with graph dependency from different context"
                ));
            }
        }
        let id = self.body.borrow().nodes.len() as u64;
        let result = Node {
            body: Arc::new(AtomicRefCell::new(NodeBody {
                graph: self.downgrade(),
                node_dependencies: node_dependencies.iter().map(|n| n.downgrade()).collect(),
                graph_dependencies: graph_dependencies.iter().map(|g| g.downgrade()).collect(),
                operation,
                id,
            })),
        };
        {
            let mut cell = self.body.borrow_mut();
            cell.nodes.push(result.clone());
        }
        let mut context_has_type_checker = false;
        {
            let context = self.get_context();
            let mut context_cell = context.body.borrow_mut();
            let type_checker = &mut context_cell.type_checker;
            if type_checker.is_some() {
                context_has_type_checker = true;
            }
        }
        if context_has_type_checker {
            let type_checking_result = result.get_type();
            if type_checking_result.is_err() {
                self.remove_last_node(result)?;
                return Err(type_checking_result.expect_err("Should not be here"));
            }
            let type_result = type_checking_result?;

            let size_estimate = get_size_estimation_in_bits(type_result);
            if size_estimate.is_err() {
                self.remove_last_node(result)?;
                return Err(runtime_error!("Trying to add a node with invalid size"));
            }
            if size_estimate? > type_size_limit_constants::MAX_INDIVIDUAL_NODE_SIZE {
                self.remove_last_node(result)?;
                return Err(runtime_error!(
                    "Trying to add a node larger than MAX_INDIVIDUAL_NODE_SIZE"
                ));
            }

            let context = self.get_context();
            let size_checking_result = context.try_update_total_size(result.clone());
            if size_checking_result.is_err() {
                self.remove_last_node(result)?;
                return Err(size_checking_result.expect_err("Should not be here"));
            }
        }
        Ok(result)
    }

    fn remove_last_node(&self, n: Node) -> Result<()> {
        if n.get_graph() != *self {
            return Err(runtime_error!(
                "The node to be removed from a different graph"
            ));
        }
        {
            let cell = self.body.borrow();
            if n != *cell
                .nodes
                .last()
                .ok_or_else(|| runtime_error!("Nodes list is empty"))?
            {
                return Err(runtime_error!(
                    "The node to be removed is not the last node"
                ));
            }
        };
        let context = self.get_context();
        context.unregister_node(n.clone())?;
        let mut context_body = context.body.borrow_mut();
        if let Some(tc) = &mut context_body.type_checker {
            tc.unregister_node(n)?;
        }
        let mut cell = self.body.borrow_mut();
        cell.nodes.pop();
        Ok(())
    }

    pub(crate) fn nop(&self, a: Node) -> Result<Node> {
        self.add_node(vec![a], vec![], Operation::NOP)
    }

    pub(crate) fn prf(&self, key: Node, iv: u64, output_type: Type) -> Result<Node> {
        self.add_node(vec![key], vec![], Operation::PRF(iv, output_type))
    }

    pub(crate) fn permutation_from_prf(&self, key: Node, iv: u64, n: u64) -> Result<Node> {
        self.add_node(vec![key], vec![], Operation::PermutationFromPRF(iv, n))
    }

    pub(super) fn is_finalized(&self) -> bool {
        self.body.borrow().finalized
    }

    pub(super) fn check_finalized(&self) -> Result<()> {
        if !self.is_finalized() {
            return Err(runtime_error!("Graph is not finalized"));
        }
        Ok(())
    }

    fn make_serializable(&self) -> SerializableGraph {
        let output_node = match self.get_output_node() {
            Ok(n) => Some(n.get_id()),
            Err(_) => None,
        };
        Arc::new(SerializableGraphBody {
            finalized: self.is_finalized(),
            nodes: self
                .get_nodes()
                .iter()
                .map(|n| n.make_serializable())
                .collect(),
            output_node,
        })
    }

    fn downgrade(&self) -> WeakGraph {
        WeakGraph {
            body: Arc::downgrade(&self.body),
        }
    }

    #[doc(hidden)]
    pub fn add_annotation(&self, annotation: GraphAnnotation) -> Result<Graph> {
        self.get_context().add_graph_annotation(self, annotation)?;
        Ok(self.clone())
    }

    pub fn get_annotations(&self) -> Result<Vec<GraphAnnotation>> {
        self.get_context().get_graph_annotations(self.clone())
    }

    /// Rearrange given input values according to the names and the order of the related input nodes.
    ///
    /// For example, given a graph with the first input node named 'A' and the second one named 'B' and input values `{'B': v, 'A': w}`, this function returns a vector `[w, v]`.
    ///
    /// # Arguments
    ///
    /// `values` - hashmap of values keyed by node names
    ///
    /// # Returns
    ///
    /// Vector of values arranged by node names
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// # use std::collections::HashMap;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n1 = g.input(t.clone()).unwrap();
    /// n1.set_name("input1").unwrap();
    /// let n2 = g.input(t.clone()).unwrap();
    /// n2.set_name("input2").unwrap();
    ///
    /// let mut input_map = HashMap::new();
    /// input_map.insert("input2", 2);
    /// input_map.insert("input1", 1);
    /// let ordered_input = g.prepare_input_values(input_map).unwrap();
    ///
    /// assert_eq!(vec![1,2], ordered_input);
    /// ```
    pub fn prepare_input_values<T: Clone>(&self, values: HashMap<&str, T>) -> Result<Vec<T>> {
        self.get_context()
            .prepare_input_values(self.clone(), values)
    }
}
type WeakGraphBodyPointer = Weak<AtomicRefCell<GraphBody>>;

struct WeakGraph {
    body: WeakGraphBodyPointer,
}

impl WeakGraph {
    //upgrade function panics if the the Graph pointer it downgraded from went out of scope
    fn upgrade(&self) -> Graph {
        Graph {
            body: self.body.upgrade().unwrap(),
        }
    }
}
impl Clone for WeakGraph {
    fn clone(&self) -> Self {
        WeakGraph {
            body: self.body.clone(),
        }
    }
}

#[doc(hidden)]
/// Various node-related properties which aren't used in the graph building
/// or type inference, but can be used in node expansion or MPC compilation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeAnnotation {
    AssociativeOperation,
    Private,
    Send(u64, u64), // (sender_index, receiver_index); indices belong to the set 0..PARTIES
    PRFMultiplication,
    PRFB2A,
    PRFTruncate,
}

#[doc(hidden)]
/// Various graph-related properties which aren't used in the graph building
/// or type inference, but can be used in node expansion or MPC compilation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphAnnotation {
    AssociativeOperation,
    OneBitState,
    SmallState,
}

struct ContextBody {
    finalized: bool,
    graphs: Vec<Graph>,
    main_graph: Option<WeakGraph>,
    /// graph_id -> name
    graphs_names: HashMap<u64, String>,
    /// name -> graph_id
    graphs_names_inverse: HashMap<String, u64>,
    /// (graph_id, node_id) -> name
    nodes_names: HashMap<(u64, u64), String>,
    /// graph_id -> (name -> node_id)
    nodes_names_inverse: HashMap<u64, HashMap<String, u64>>,
    /// (graph_id, node_id) -> NodeAnnotation's
    nodes_annotations: HashMap<(u64, u64), Vec<NodeAnnotation>>,
    /// (graph_id) -> GraphAnnotation's
    graphs_annotations: HashMap<u64, Vec<GraphAnnotation>>,
    total_size_nodes: u64,
    type_checker: Option<TypeInferenceWorker>,
}

type ContextBodyPointer = Arc<AtomicRefCell<ContextBody>>;

/// A structure that stores a pointer to a computation context that contains related computation graphs.
///
/// Context is a basic object to create computation graphs, arrange data flow between them and keep necessary information about them that is used for optimization, secure compilation and evaluation.
///
/// Context should have a main graph and be finalized in order to evaluate any of its graphs.
///
/// # Rust crates
///
/// [Clone] trait duplicates the pointer, not the underlying context.
///
/// [PartialEq] trait compares pointers, not the related contexts.
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate maplit;
/// # fn main() {
/// # use ciphercore_base::graphs::{Context, create_context};
/// # use ciphercore_base::data_values::Value;
/// # use ciphercore_base::evaluators::random_evaluate;
/// # use ciphercore_base::data_types::{INT32, scalar_type};
/// # use ciphercore_base::errors::Result;
/// let context = || -> Result<Context> {
///     let context = create_context()?;
///     let graph = context.create_graph()?.set_name("main")?;
///     graph
///         .input(scalar_type(INT32))?
///         .set_name("a")?
///         .add(graph
///             .input(scalar_type(INT32))?
///             .set_name("b")?)?
///         .set_as_output()?;
///     graph.finalize()?.set_as_main()?;
///     context.finalize()?;
///     Ok(context)
/// }().unwrap();
///
/// let result = || -> Result<i32> {
///     let g = context.retrieve_graph("main")?;
///     let result = random_evaluate(
///         g.clone(),
///         g.prepare_input_values(
///             hashmap!{
///                 "a" => Value::from_scalar(123, INT32)?,
///                 "b" => Value::from_scalar(654, INT32)?,
///             },
///         )?,
///     )?;
///     let result = result.to_i32(INT32)?;
///     Ok(result)
/// }().unwrap();
///
/// assert_eq!(result, 777);
/// # }
/// ```
#[cfg_attr(feature = "py-binding", struct_wrapper)]
pub struct Context {
    body: ContextBodyPointer,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("body", &self.body.as_ptr())
            .finish()
    }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match serde_json::to_string(&self) {
            Ok(s) => write!(f, "{s}"),
            Err(_err) => Err(fmt::Error::default()),
        }
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Graph[num_nodes={}]", self.get_num_nodes())
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Node[type={}]", self.get_type()?)
    }
}

#[derive(Serialize, Deserialize)]
struct SerializableContextBody {
    finalized: bool,
    graphs: Vec<SerializableGraph>,
    main_graph: Option<u64>,
    /// graph_id -> name
    graphs_names: Vec<(u64, String)>,
    /// (graph_id, node_id) -> name
    nodes_names: Vec<((u64, u64), String)>,
    /// (graph_id, node_id) -> NodeAnnotation's
    nodes_annotations: Vec<((u64, u64), Vec<NodeAnnotation>)>,
    /// (graph_id) -> GraphAnnotation's
    graphs_annotations: Vec<(u64, Vec<GraphAnnotation>)>,
}

impl SerializableContextBody {
    fn recover_original_graph(
        serializable_graph: SerializableGraph,
        context: Context,
    ) -> Result<Graph> {
        let result_graph = context.create_graph()?;
        for node in &serializable_graph.nodes {
            let mut node_dependencies = vec![];
            for id in &node.node_dependencies {
                let current_nodes = &result_graph.body.borrow().nodes;
                if *id >= current_nodes.len() as u64 {
                    return Err(runtime_error!("Non-existent node dependency"));
                }
                node_dependencies.push(current_nodes[*id as usize].clone());
            }
            let mut graph_dependencies = vec![];
            for id in &node.graph_dependencies {
                let context = result_graph.get_context();
                let current_graphs = &context.body.borrow().graphs;
                if *id >= current_graphs.len() as u64 {
                    return Err(runtime_error!("Non-existent graph dependency"));
                }
                graph_dependencies.push(current_graphs[*id as usize].clone());
            }
            result_graph.add_node(
                node_dependencies,
                graph_dependencies,
                node.operation.clone(),
            )?;
        }
        if let Some(id) = serializable_graph.output_node {
            let rebuilt_output_node = {
                let current_nodes = &result_graph.body.borrow().nodes;
                if id >= current_nodes.len() as u64 {
                    return Err(runtime_error!("Non-existent output node"));
                }
                current_nodes[id as usize].clone()
            };
            result_graph.set_output_node(rebuilt_output_node)?;
        }
        if serializable_graph.finalized {
            result_graph.finalize()?;
        }
        Ok(result_graph)
    }

    fn recover_original_context(&self) -> Result<Context> {
        let result_context = create_context()?;
        for graph in &self.graphs {
            let _result_graph =
                Self::recover_original_graph(graph.clone(), result_context.clone())?;
        }
        if let Some(id) = self.main_graph {
            let rebuilt_main_graph = {
                let current_graphs = &result_context.body.borrow().graphs;
                if id >= current_graphs.len() as u64 {
                    return Err(runtime_error!("Non-existent main graph"));
                }
                current_graphs[id as usize].clone()
            };
            result_context.set_main_graph(rebuilt_main_graph)?;
        }
        for (id, _) in &self.graphs_names {
            let current_graphs = &result_context.body.borrow().graphs;
            if *id >= current_graphs.len() as u64 {
                return Err(runtime_error!("graphs_names contain an invalid ID"));
            }
        }
        for ((graph_id, node_id), _) in &self.nodes_names {
            let current_graphs = &result_context.body.borrow().graphs;
            if *graph_id >= current_graphs.len() as u64 {
                return Err(runtime_error!("nodes_names contain an invalid graph ID"));
            }
            let current_nodes = &current_graphs[*graph_id as usize].body.borrow().nodes;
            if *node_id >= current_nodes.len() as u64 {
                return Err(runtime_error!("nodes_names contain an invalid node ID"));
            }
        }
        for (id, name) in &self.graphs_names {
            let current_graph = {
                let current_graphs = &result_context.body.borrow().graphs;
                current_graphs[*id as usize].clone()
            };
            result_context.set_graph_name(current_graph, name)?;
        }
        for ((graph_id, node_id), name) in &self.nodes_names {
            let current_node = {
                let current_graphs = &result_context.body.borrow().graphs;
                let current_nodes = &current_graphs[*graph_id as usize].body.borrow().nodes;
                current_nodes[*node_id as usize].clone()
            };
            result_context.set_node_name(current_node, name)?;
        }
        for (id, annotations) in &self.graphs_annotations {
            let current_graph = {
                let current_graphs = &result_context.body.borrow().graphs;
                current_graphs[*id as usize].clone()
            };
            for annotation in annotations {
                result_context.add_graph_annotation(&current_graph, annotation.clone())?;
            }
        }
        for ((graph_id, node_id), annotations) in &self.nodes_annotations {
            let current_node = {
                let current_graphs = &result_context.body.borrow().graphs;
                let current_nodes = &current_graphs[*graph_id as usize].body.borrow().nodes;
                current_nodes[*node_id as usize].clone()
            };
            for annotation in annotations {
                result_context.add_node_annotation(&current_node, annotation.clone())?;
            }
        }
        if self.finalized {
            result_context.finalize()?;
        }
        Ok(result_context)
    }
}

type SerializableContext = Arc<SerializableContextBody>;

impl Clone for Context {
    /// Returns a new [Context] value with a copy of the pointer to a node.
    fn clone(&self) -> Self {
        Context {
            body: self.body.clone(),
        }
    }
}

impl PartialEq for Context {
    /// Tests whether `self` and `other` contexts are equal via comparison of their respective pointers.
    ///
    /// # Arguments
    ///
    /// `other` - another [Context] value
    ///
    /// # Returns
    ///
    /// `true` if `self` and `other` are equal, `false` otherwise
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.body, &other.body)
    }
}

impl Eq for Context {}

/// Public methods which supposed to be imported in Python.
#[cfg_attr(feature = "py-binding", impl_wrapper)]
impl Context {
    /// Creates an empty computation graph in this context.
    ///
    /// # Returns
    ///
    /// New computation graph
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// ```
    pub fn create_graph(&self) -> Result<Graph> {
        if self.body.borrow().finalized {
            return Err(runtime_error!("Can't add a graph to a finalized context"));
        }
        let id = self.body.borrow().graphs.len() as u64;
        let result = Graph {
            body: Arc::new(AtomicRefCell::new(GraphBody {
                finalized: false,
                nodes: vec![],
                output_node: None,
                id,
                context: self.downgrade(),
            })),
        };
        self.body.borrow_mut().graphs.push(result.clone());
        Ok(result)
    }

    /// Finalizes the context if all its graphs are finalized and the main graph is set.
    ///
    /// After finalization the context can't be changed.
    ///
    /// # Returns
    ///
    /// Finalized context
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, vector_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let vec_t = vector_type(4, t);
    /// let n1 = g.input(vec_t).unwrap();
    /// let n2 = g.vector_to_array(n1).unwrap();
    /// n2.set_as_output().unwrap();
    /// g.finalize().unwrap();
    /// c.set_main_graph(g).unwrap();
    /// c.finalize().unwrap();
    /// ```
    pub fn finalize(&self) -> Result<Context> {
        for graph in self.get_graphs() {
            graph.check_finalized()?;
        }
        let main_graph = self.body.borrow().main_graph.clone();
        match main_graph {
            Some(_) => {
                self.body.borrow_mut().finalized = true;
                Ok(self.clone())
            }
            _ => Err(runtime_error!(
                "Can't finalize the context without the main graph"
            )),
        }
    }

    /// Promotes a graph to the main one in this context.
    ///
    /// # Arguments
    ///
    /// `graph` - graph
    ///
    /// # Returns
    ///
    /// This context
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = array_type(vec![3, 2], INT32);
    /// let n = g.input(t).unwrap();
    /// n.set_as_output().unwrap();
    /// g.finalize().unwrap();
    /// c.set_main_graph(g).unwrap();
    /// ```
    pub fn set_main_graph(&self, graph: Graph) -> Result<Context> {
        let current_main_graph = self.body.borrow().main_graph.clone();
        match current_main_graph {
            Some(_) => Err(runtime_error!("Main graph is already set")),
            None => {
                if graph.get_context() != *self {
                    return Err(runtime_error!("Main graph is from the wrong context"));
                }
                graph.check_finalized()?;
                self.body.borrow_mut().main_graph = Some(graph.downgrade());
                Ok(self.clone())
            }
        }
    }

    /// Returns the vector of graphs contained in this context in order of creation.
    ///
    /// # Returns
    ///
    /// Vector of the graphs in this context
    pub fn get_graphs(&self) -> Vec<Graph> {
        self.body.borrow().graphs.clone()
    }

    /// Does nothing if the context is finalized; otherwise returns a runtime error.
    ///
    /// # Returns
    ///
    /// Runtime error if this context is not finalized
    pub fn check_finalized(&self) -> Result<()> {
        if !self.is_finalized() {
            return Err(runtime_error!("Context is not finalized"));
        }
        Ok(())
    }

    /// Returns the main graph of the context if it is already set.
    ///
    /// # Returns
    ///
    /// Main graph of the context
    pub fn get_main_graph(&self) -> Result<Graph> {
        match &self.body.borrow().main_graph {
            Some(g) => Ok(g.upgrade()),
            None => Err(runtime_error!("main graph is not set")),
        }
    }

    /// Returns the number of graphs contained in this context.
    ///
    /// # Returns
    ///
    /// Number of the graphs in this context
    pub fn get_num_graphs(&self) -> u64 {
        self.body.borrow().graphs.len() as u64
    }

    /// Returns the graph contained in this context with a given ID.
    ///
    /// A graph ID is a serial number of a graph between `0` and `n-1` where `n` is the number of graphs in this context.
    ///
    /// # Arguments
    ///
    /// `id` - ID of a graph
    ///
    /// # Returns
    ///
    /// Graph with the given ID
    pub fn get_graph_by_id(&self, id: u64) -> Result<Graph> {
        let graphs = &self.body.borrow().graphs;
        if id >= graphs.len() as u64 {
            Err(runtime_error!("Invalid id for the graph retrieval"))
        } else {
            Ok(graphs[id as usize].clone())
        }
    }

    /// Returns the node contained in this context with a given global ID.
    ///
    /// The global ID of a node is a pair of the node ID and the ID of its parent graph.
    ///
    /// # Arguments
    ///
    /// `id` - tuple (graph ID, node ID)
    ///
    /// # Returns
    ///
    /// Node with the given global ID
    pub fn get_node_by_global_id(&self, id: (u64, u64)) -> Result<Node> {
        self.get_graph_by_id(id.0)?.get_node_by_id(id.1)
    }

    /// Sets the name of a graph.
    ///
    /// A given name should be unique.
    ///
    /// # Arguments
    ///
    /// * `graph` - graph
    /// * `name` - name of the graph
    ///
    /// # Returns
    ///
    /// This context
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// g.set_name("relu").unwrap();
    /// ```
    pub fn set_graph_name(&self, graph: Graph, name: &str) -> Result<Context> {
        if graph.get_context() != *self {
            return Err(runtime_error!(
                "The graph to be named is in a different context"
            ));
        }
        if self.is_finalized() {
            return Err(runtime_error!(
                "Can't set a graph name in a finalized context"
            ));
        }
        let id = graph.get_id();
        let name_owned = name.to_owned();
        let mut cell = self.body.borrow_mut();
        if cell.graphs_names.get(&id).is_some() {
            return Err(runtime_error!("Can't set the graph name twice"));
        }
        if cell.graphs_names_inverse.get(name).is_some() {
            return Err(runtime_error!("Graph names must be unique"));
        }
        cell.graphs_names.insert(id, name_owned.clone());
        cell.graphs_names_inverse.insert(name_owned, id);
        Ok(self.clone())
    }

    /// Returns the name of a graph.
    ///
    /// # Arguments
    ///
    /// `graph` - graph
    ///
    /// # Returns
    ///
    /// Name of a given graph
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// g.set_name("relu").unwrap();
    /// assert_eq!(c.get_graph_name(g).unwrap(), "relu".to_owned());
    /// ```
    pub fn get_graph_name(&self, graph: Graph) -> Result<String> {
        if graph.get_context() != *self {
            return Err(runtime_error!("The graph is in a different context"));
        }
        let cell = self.body.borrow();
        Ok(cell
            .graphs_names
            .get(&graph.get_id())
            .ok_or_else(|| runtime_error!("The graph does not have a name assigned"))?
            .clone())
    }

    /// Returns the graph with a given name in this context.
    ///
    /// # Arguments
    ///
    /// `name` - graph name
    ///
    /// # Returns
    ///
    /// Graph with a given name
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let n = g.input(scalar_type(BIT)).unwrap();
    /// g.set_name("input_graph").unwrap();
    /// assert!(g == c.retrieve_graph("input_graph").unwrap());
    /// ```
    pub fn retrieve_graph(&self, name: &str) -> Result<Graph> {
        let cell = self.body.borrow();
        let id = cell
            .graphs_names_inverse
            .get(name)
            .ok_or_else(|| runtime_error!("No graph with such a name exists"))?;
        let graph = cell.graphs[*id as usize].clone();
        Ok(graph)
    }

    /// Sets the name of a node.
    ///
    /// A given name should be unique.
    ///
    /// # Arguments
    ///
    /// * `node` - node
    /// * `name` - name of a node
    ///
    /// # Returns
    ///
    /// This context
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{scalar_type, BIT};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n = g.input(t).unwrap();
    /// c.set_node_name(n, "XOR").unwrap();
    /// ```
    pub fn set_node_name(&self, node: Node, name: &str) -> Result<Context> {
        if node.get_graph().get_context() != *self {
            return Err(runtime_error!(
                "The node to be named is in a different context"
            ));
        }
        if self.is_finalized() {
            return Err(runtime_error!(
                "Can't set a node name in a finalized context"
            ));
        }
        let node_id = node.get_id();
        let graph_id = node.get_graph().get_id();
        let mut cell = self.body.borrow_mut();
        if cell.nodes_names.get(&(graph_id, node_id)).is_some() {
            return Err(runtime_error!("Can't set the node name twice"));
        }
        if cell.nodes_names_inverse.get(&graph_id).is_none() {
            cell.nodes_names_inverse.insert(graph_id, HashMap::new());
        }
        let graph_map_inverse = cell
            .nodes_names_inverse
            .get_mut(&graph_id)
            .expect("Should not be here!");
        if graph_map_inverse.get(name).is_some() {
            return Err(runtime_error!(
                "Node names must be unique (within the graph)"
            ));
        }
        graph_map_inverse.insert(name.to_owned(), node_id);
        cell.nodes_names
            .insert((graph_id, node_id), name.to_owned());
        Ok(self.clone())
    }

    /// Returns the name of a node.
    ///
    /// # Arguments
    ///
    /// `node` - node
    ///
    /// # Returns
    ///
    /// Name of a node or None if it doesn't have a name
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{scalar_type, BIT};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let n = g.input(t).unwrap();
    /// n.set_name("XOR").unwrap();
    /// assert_eq!(c.get_node_name(n).unwrap(), Some("XOR".to_owned()));
    /// ```
    pub fn get_node_name(&self, node: Node) -> Result<Option<String>> {
        if node.get_graph().get_context() != *self {
            return Err(runtime_error!("The node is in a different context"));
        }
        let node_id = node.get_id();
        let graph_id = node.get_graph().get_id();
        let cell = self.body.borrow();
        Ok(cell.nodes_names.get(&(graph_id, node_id)).cloned())
    }

    /// Returns the node with a given name in a given graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - graph
    /// * `name` - node name
    ///
    /// # Returns
    ///
    /// Node with a given name
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::create_context;
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let n = g.input(scalar_type(BIT)).unwrap();
    /// n.set_name("input_node").unwrap();
    /// assert!(n == c.retrieve_node(g, "input_node").unwrap());
    /// ```
    pub fn retrieve_node(&self, graph: Graph, name: &str) -> Result<Node> {
        if graph.get_context() != *self {
            return Err(runtime_error!("The graph is in a different context"));
        }
        let graph_id = graph.get_id();
        let cell = self.body.borrow();
        let node_id = cell
            .nodes_names_inverse
            .get(&graph_id)
            .ok_or_else(|| runtime_error!("The graph has no named nodes"))?
            .get(name)
            .ok_or_else(|| runtime_error!("Node with a given name does not exist"))?;
        Ok(graph.body.borrow().nodes[*node_id as usize].clone())
    }
    /// Check that two given contexts contain the same data, i.e. graphs, nodes, names, parameters.
    ///
    /// Underlying structures that contain pointers (graphs, nodes) are compared by data they refer to.
    ///
    /// # Arguments
    ///
    /// * `context2` - context to compare
    ///
    /// # Returns
    ///
    /// `true` if the given contexts contain the same content, otherwise `false`
    pub fn deep_equal(&self, context2: Context) -> bool {
        contexts_deep_equal(self.clone(), context2)
    }
}

fn serialize_hashmap<K, V>(map: HashMap<K, V>) -> Vec<(K, V)>
where
    K: Ord + Copy,
{
    let mut vec: Vec<_> = map.into_iter().collect();
    vec.sort_by_key(|(k, _)| *k);
    vec
}

/// Methods which aren't supposed to be imported in Python.
impl Context {
    pub(super) fn is_finalized(&self) -> bool {
        self.body.borrow().finalized
    }

    fn make_serializable(&self) -> SerializableContext {
        let main_graph = match self.get_main_graph() {
            Ok(g) => Some(g.get_id()),
            Err(_) => None,
        };
        let cell = self.body.borrow();
        Arc::new(SerializableContextBody {
            finalized: self.is_finalized(),
            graphs: self
                .get_graphs()
                .iter()
                .map(|g| g.make_serializable())
                .collect(),
            main_graph,
            graphs_names: serialize_hashmap(cell.graphs_names.clone()),
            nodes_names: serialize_hashmap(cell.nodes_names.clone()),
            graphs_annotations: serialize_hashmap(cell.graphs_annotations.clone()),
            nodes_annotations: serialize_hashmap(cell.nodes_annotations.clone()),
        })
    }

    fn add_type_checker(&self) -> Result<Context> {
        {
            let mut cell = self.body.borrow_mut();
            if cell.type_checker.is_some() {
                return Err(runtime_error!(
                    "Type checker associated with the context already exists"
                ));
            }
            cell.type_checker = Some(create_type_inference_worker(self.clone()));
        }
        for graph in self.get_graphs() {
            for node in graph.get_nodes() {
                node.get_type()?;
            }
        }
        Ok(self.clone())
    }

    fn get_total_size_nodes(&self) -> u64 {
        self.body.borrow().total_size_nodes
    }

    fn set_total_size_nodes(&self, size: u64) {
        self.body.borrow_mut().total_size_nodes = size;
    }

    fn try_update_total_size(&self, node: Node) -> Result<()> {
        let node_type = match node.get_operation() {
            Operation::Input(input_type) => input_type,
            Operation::Constant(t, _) => t,
            _ => return Ok(()),
        };
        if !node_type.is_valid() {
            return Err(runtime_error!("Node with an invalid type: {:?}", node_type));
        }
        let new_total_size = self
            .get_total_size_nodes()
            .checked_add(get_size_estimation_in_bits(node_type)?)
            .ok_or_else(|| runtime_error!("add overflow!"))?;
        if new_total_size > type_size_limit_constants::MAX_TOTAL_SIZE_NODES {
            return Err(runtime_error!(
                "Can't add a node: total size of nodes exceeds MAX_TOTAL_SIZE_NODES"
            ));
        }
        self.set_total_size_nodes(new_total_size);
        Ok(())
    }

    fn unregister_node(&self, node: Node) -> Result<()> {
        if node.get_graph().get_context() != *self {
            return Err(runtime_error!(
                "The node to be unregister from  a different context"
            ));
        }
        if self.is_finalized() {
            return Err(runtime_error!(
                "Can't unregister a node from  a finalized context"
            ));
        }

        let node_id = node.get_id();
        let graph_id = node.get_graph().get_id();

        let mut cell = self.body.borrow_mut();
        let name_option = cell.nodes_names.remove(&(graph_id, node_id));
        cell.nodes_annotations.remove(&(graph_id, node_id));
        if cell.nodes_names_inverse.get(&graph_id).is_none() {
            return Ok(());
        }
        let graph_map_inverse = cell
            .nodes_names_inverse
            .get_mut(&graph_id)
            .expect("Should not be here!");
        if let Some(name) = name_option {
            graph_map_inverse.remove(&name);
        }
        Ok(())
    }

    fn to_versioned_data(&self) -> Result<VersionedData> {
        VersionedData::create_versioned_data(
            DATA_VERSION,
            serde_json::to_string(&self.make_serializable())?,
        )
    }
    fn prepare_input_values<T: Clone>(
        &self,
        graph: Graph,
        values: HashMap<&str, T>,
    ) -> Result<Vec<T>> {
        if graph.get_context() != *self {
            return Err(runtime_error!("The graph is in a different context"));
        }
        let graph_id = graph.get_id();
        let cell = self.body.borrow();
        for node_name in values.keys() {
            cell.nodes_names_inverse
                .get(&graph_id)
                .ok_or_else(|| runtime_error!("Trying to call graph without named nodes"))?
                .get(node_name as &str)
                .ok_or_else(|| runtime_error!("Input with a given name is not found"))?;
        }
        let mut result = vec![];
        for node in graph.get_nodes() {
            if node.get_operation().is_input() {
                let node_id = node.get_id();
                let node_name = cell
                    .nodes_names
                    .get(&(graph_id, node_id))
                    .ok_or_else(|| runtime_error!("Unnamed input"))?;
                let node_value = values
                    .get(node_name as &str)
                    .ok_or_else(|| runtime_error!("Unspecified input"))?
                    .clone();
                result.push(node_value);
            }
        }
        Ok(result)
    }

    pub(super) fn add_node_annotation(
        &self,
        node: &Node,
        annotation: NodeAnnotation,
    ) -> Result<Context> {
        if node.get_graph().get_context() != *self {
            return Err(runtime_error!(
                "The node to be annotated is in a different context"
            ));
        }
        if self.is_finalized() {
            return Err(runtime_error!(
                "Can't add a node annotation in a finalized context"
            ));
        }
        let node_id = node.get_id();
        let graph_id = node.get_graph().get_id();
        let key = (graph_id, node_id);
        let mut cell = self.body.borrow_mut();
        let annotations = cell.nodes_annotations.get_mut(&key);
        if let Some(annotation_vec) = annotations {
            annotation_vec.push(annotation);
        } else {
            cell.nodes_annotations.insert(key, vec![annotation]);
        }
        Ok(self.clone())
    }

    pub(super) fn get_node_annotations(&self, node: Node) -> Result<Vec<NodeAnnotation>> {
        if node.get_graph().get_context() != *self {
            return Err(runtime_error!("The node is in a different context"));
        }
        let node_id = node.get_id();
        let graph_id = node.get_graph().get_id();
        let cell = self.body.borrow();
        Ok(cell
            .nodes_annotations
            .get(&(graph_id, node_id))
            .cloned()
            .unwrap_or_default())
    }

    fn add_graph_annotation(&self, graph: &Graph, annotation: GraphAnnotation) -> Result<Context> {
        if graph.get_context() != *self {
            return Err(runtime_error!(
                "The graph to be annotated is in a different context"
            ));
        }
        if self.is_finalized() {
            return Err(runtime_error!(
                "Can't set a graph annotation in a finalized context"
            ));
        }
        let id = graph.get_id();
        let mut cell = self.body.borrow_mut();
        let annotations = cell.graphs_annotations.get_mut(&id);
        if let Some(annotation_vec) = annotations {
            annotation_vec.push(annotation);
        } else {
            cell.graphs_annotations.insert(id, vec![annotation]);
        }
        Ok(self.clone())
    }

    fn get_graph_annotations(&self, graph: Graph) -> Result<Vec<GraphAnnotation>> {
        if graph.get_context() != *self {
            return Err(runtime_error!("The graph is in a different context"));
        }
        let cell = self.body.borrow();
        Ok(cell
            .graphs_annotations
            .get(&graph.get_id())
            .cloned()
            .unwrap_or_default())
    }

    pub(super) fn downgrade(&self) -> WeakContext {
        WeakContext {
            body: Arc::downgrade(&self.body),
        }
    }
}

impl Serialize for Context {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let versioned_context = self
            .to_versioned_data()
            .expect("Error during conversion from Context into VersionedData");
        //VersionedData::from(self.clone());
        versioned_context.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Context {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Context, D::Error>
    where
        D: Deserializer<'de>,
    {
        let versioned_context = VersionedData::deserialize(deserializer)?;
        if !versioned_context.check_version(DATA_VERSION) {
            Err(runtime_error!(
                "Context version doesn't match the requirement"
            ))
            .map_err(serde::de::Error::custom)
        } else {
            let serializable_context =
                serde_json::from_str::<SerializableContext>(versioned_context.get_data_string())
                    .expect("Error during deserialization of SerializableContext");
            serializable_context
                .recover_original_context()
                .map_err(serde::de::Error::custom)
        }
    }
}

/// In general, `create_unchecked_context()` should not return errors, but
/// we still make the result type Result<Context> for uniformity.
pub(super) fn create_unchecked_context() -> Result<Context> {
    Ok(Context {
        body: Arc::new(AtomicRefCell::new(ContextBody {
            finalized: false,
            graphs: vec![],
            main_graph: None,
            graphs_names: HashMap::new(),
            graphs_names_inverse: HashMap::new(),
            nodes_names: HashMap::new(),
            nodes_names_inverse: HashMap::new(),
            graphs_annotations: HashMap::new(),
            nodes_annotations: HashMap::new(),
            type_checker: None,
            total_size_nodes: 0,
        })),
    })
}

/// Creates an empty computation context.
///
/// # Returns
///
/// New computation context
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// let c = create_context().unwrap();
/// ```
#[cfg_attr(feature = "py-binding", fn_wrapper)]
pub fn create_context() -> Result<Context> {
    let context = create_unchecked_context()?;
    context.add_type_checker()?;
    Ok(context)
}

fn graphs_deep_equal(graph1: Graph, graph2: Graph) -> bool {
    let graph1_body = graph1.body.borrow();
    let graph2_body = graph2.body.borrow();
    if graph1_body.finalized != graph2_body.finalized {
        return false;
    }
    if graph1_body.nodes.len() != graph2_body.nodes.len() {
        return false;
    }
    for j in 0..graph1_body.nodes.len() {
        let node1 = graph1_body.nodes[j].clone();
        let node2 = graph2_body.nodes[j].clone();
        let node1_body = node1.body.borrow();
        let node2_body = node2.body.borrow();
        if node1_body.operation != node2_body.operation {
            return false;
        }
        let node_dependencies1: Vec<u64> = node1_body
            .node_dependencies
            .iter()
            .map(|n| n.upgrade().get_id())
            .collect();
        let node_dependencies2: Vec<u64> = node2_body
            .node_dependencies
            .iter()
            .map(|n| n.upgrade().get_id())
            .collect();
        if node_dependencies1 != node_dependencies2 {
            return false;
        }
        let graph_dependencies1: Vec<u64> = node1_body
            .graph_dependencies
            .iter()
            .map(|g| g.upgrade().get_id())
            .collect();
        let graph_dependencies2: Vec<u64> = node2_body
            .graph_dependencies
            .iter()
            .map(|g| g.upgrade().get_id())
            .collect();
        if graph_dependencies1 != graph_dependencies2 {
            return false;
        }
    }
    if graph1_body
        .output_node
        .clone()
        .map(|n| n.upgrade().get_id())
        != graph2_body
            .output_node
            .clone()
            .map(|n| n.upgrade().get_id())
    {
        return false;
    }
    true
}

/// Check that two given contexts contain the same data, i.e. graphs, nodes, names, parameters.
///
/// Underlying structures that contain pointers (graphs, nodes) are compared by data they refer to.
///
/// # Arguments
///
/// * `context1` - first context to compare
/// * `context2` - second context to compare
///
/// # Returns
///
/// `true` if the given contexts contain the same content, otherwise `false`
pub fn contexts_deep_equal(context1: Context, context2: Context) -> bool {
    let body1 = context1.body.borrow();
    let body2 = context2.body.borrow();
    if body1.finalized != body2.finalized {
        return false;
    }
    if body1.graphs_names != body2.graphs_names {
        return false;
    }
    if body1.nodes_names != body2.nodes_names {
        return false;
    }
    if body1.nodes_annotations != body2.nodes_annotations {
        return false;
    }
    if body1.graphs_annotations != body2.graphs_annotations {
        return false;
    }
    if body1.graphs.len() != body2.graphs.len() {
        return false;
    }
    for i in 0..body1.graphs.len() {
        if !graphs_deep_equal(body1.graphs[i].clone(), body2.graphs[i].clone()) {
            return false;
        }
    }
    body1.main_graph.clone().map(|g| g.upgrade().get_id())
        == body2.main_graph.clone().map(|g| g.upgrade().get_id())
}

// Pass the node name of `in_node` to `out_node` if it is present.
pub(crate) fn copy_node_name(in_node: Node, out_node: Node) -> Result<()> {
    if let Some(node_name) = in_node.get_name()? {
        out_node.set_name(&node_name)?;
    }
    Ok(())
}

type WeakContextBodyPointer = Weak<AtomicRefCell<ContextBody>>;

pub(super) struct WeakContext {
    body: WeakContextBodyPointer,
}

impl WeakContext {
    //upgrade function panics if the the Context pointer it downgraded from went out of scope
    pub(super) fn upgrade(&self) -> Context {
        Context {
            body: self.body.upgrade().unwrap(),
        }
    }
}

#[doc(hidden)]
#[cfg(feature = "py-binding")]
#[pyo3::pymethods]
impl PyBindingSliceElement {
    #[staticmethod]
    pub fn from_single_element(ind: i64) -> Self {
        PyBindingSliceElement {
            inner: SliceElement::SingleIndex(ind),
        }
    }
    #[staticmethod]
    pub fn from_sub_array(start: Option<i64>, end: Option<i64>, step: Option<i64>) -> Self {
        PyBindingSliceElement {
            inner: SliceElement::SubArray(start, end, step),
        }
    }
    #[staticmethod]
    pub fn from_ellipsis() -> Self {
        PyBindingSliceElement {
            inner: SliceElement::Ellipsis,
        }
    }
}

pub mod util {
    use super::{create_context, Context, Graph, Node};
    use crate::errors::Result;

    /// Creates a computation context with a single graph within it.
    ///
    /// The graph is passed to the provided `build_graph_fn` function to
    /// specify the computation. The node returned by the function is marked
    /// as output.
    ///
    /// # Returns
    ///
    /// New computation context
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::util::simple_context;
    /// # use ciphercore_base::data_types::{scalar_type, INT32};
    /// let c = simple_context(|g| {
    ///     let a = g.input(scalar_type(INT32))?;
    ///     let b = g.input(scalar_type(INT32))?;
    ///     g.add(a, b)
    /// }).unwrap();
    /// ```
    pub fn simple_context<F>(build_graph_fn: F) -> Result<Context>
    where
        F: FnOnce(&Graph) -> Result<Node>,
    {
        let c = create_context()?;
        let g = c.create_graph()?;
        let out = build_graph_fn(&g)?;
        out.set_as_output()?;
        g.finalize()?;
        g.set_as_main()?;
        c.finalize()?;
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{
        array_type, scalar_type, tuple_type, vector_type, BIT, UINT16, UINT64,
    };
    use crate::inline::inline_ops::InlineConfig;
    use crate::mpc::mpc_compiler::{prepare_for_mpc_evaluation, IOStatus};
    use crate::version::DATA_VERSION;
    use std::panic;
    use std::rc::Rc;

    #[test]
    fn test_wellformed_cases() {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let input1 = graph.input(scalar_type(BIT)).unwrap();
        let input2 = graph.input(scalar_type(BIT)).unwrap();
        graph.add(input1.clone(), input2.clone()).unwrap();
        graph.subtract(input1.clone(), input2.clone()).unwrap();
        graph.multiply(input1.clone(), input2.clone()).unwrap();
        graph.dot(input1.clone(), input2.clone()).unwrap();
        graph.matmul(input1.clone(), input2.clone()).unwrap();
        graph.truncate(input1.clone(), 123).unwrap();
        let input3 = graph.input(array_type(vec![10, 20, 30], BIT)).unwrap();
        graph.sum(input3.clone(), vec![1, 2]).unwrap();
        graph.permute_axes(input3.clone(), vec![1, 2, 0]).unwrap();
        graph.get(input3.clone(), vec![1, 2]).unwrap();
        graph
            .reshape(input3.clone(), array_type(vec![20, 300], BIT))
            .unwrap();
        graph.nop(input3.clone()).unwrap();
        let key = graph.random(array_type(vec![128], BIT)).unwrap();
        graph
            .prf(key.clone(), 0, array_type(vec![10, 10], UINT64))
            .unwrap();
        graph
            .stack(vec![input1.clone(), input2.clone()], vec![2, 1])
            .unwrap();
        let c = graph
            .constant(scalar_type(BIT), Value::from_bytes(vec![1]))
            .unwrap();
        let input4 = graph.input(array_type(vec![10, 10], UINT64)).unwrap();
        let bits = graph.a2b(input4.clone()).unwrap();
        graph.b2a(bits.clone(), UINT64).unwrap();
        let t = graph
            .create_tuple(vec![input1.clone(), input2.clone()])
            .unwrap();
        let _v = graph
            .create_vector(scalar_type(BIT), vec![input1.clone(), input2.clone()])
            .unwrap();
        let nt = graph
            .create_named_tuple(vec![
                ("Name".to_owned(), input1.clone()),
                ("Gender".to_owned(), input2.clone()),
            ])
            .unwrap();
        graph.tuple_get(t, 1).unwrap();
        graph.named_tuple_get(nt, "Gender".to_owned()).unwrap();
        let v = graph.repeat(c.clone(), 100).unwrap();
        graph.zip(vec![v.clone(), v.clone(), v.clone()]).unwrap();
        let zero = graph
            .constant(scalar_type(UINT64), Value::from_bytes(vec![0; 8]))
            .unwrap();
        graph.vector_get(v, zero).unwrap();
        graph.array_to_vector(input1.clone()).unwrap();
        graph.vector_to_array(input1.clone()).unwrap();
    }

    #[test]
    fn call_iterate_test() {
        let context = create_unchecked_context().unwrap();
        let single_bit_adder = context.create_graph().unwrap();
        {
            let carry = single_bit_adder.input(scalar_type(BIT)).unwrap();
            let inputs = single_bit_adder
                .input(tuple_type(vec![scalar_type(BIT), scalar_type(BIT)]))
                .unwrap();
            let a = single_bit_adder.tuple_get(inputs.clone(), 0).unwrap();
            let b = single_bit_adder.tuple_get(inputs.clone(), 1).unwrap();
            let ac = single_bit_adder.add(carry.clone(), a.clone()).unwrap();
            let bc = single_bit_adder.add(carry.clone(), b.clone()).unwrap();
            let result = single_bit_adder.add(ac.clone(), b.clone()).unwrap();
            let result_carry = single_bit_adder
                .add(
                    single_bit_adder.multiply(ac.clone(), bc.clone()).unwrap(),
                    carry,
                )
                .unwrap();
            let output = single_bit_adder
                .create_tuple(vec![result_carry.clone(), result.clone()])
                .unwrap();
            single_bit_adder.set_output_node(output).unwrap();
            single_bit_adder.finalize().unwrap();
        }
        let v32 = vector_type(32, scalar_type(BIT));
        let adder = context.create_graph().unwrap();
        {
            let a = adder.input(v32.clone()).unwrap();
            let b = adder.input(v32.clone()).unwrap();
            let azb = adder.zip(vec![a, b]).unwrap();
            let c = adder
                .constant(scalar_type(BIT), Value::from_bytes(vec![0]))
                .unwrap();
            let cr = adder.iterate(single_bit_adder, c, azb).unwrap();
            let r = adder.tuple_get(cr, 1).unwrap();
            adder.set_output_node(r).unwrap();
            adder.finalize().unwrap();
        }
        let three_adder = context.create_graph().unwrap();
        let a = three_adder.input(v32.clone()).unwrap();
        let b = three_adder.input(v32.clone()).unwrap();
        let c = three_adder.input(v32.clone()).unwrap();
        let result = three_adder
            .call(
                adder.clone(),
                vec![three_adder.call(adder.clone(), vec![a, b]).unwrap(), c],
            )
            .unwrap();
        three_adder.set_output_node(result).unwrap();
        three_adder.finalize().unwrap();
        context.set_main_graph(three_adder).unwrap();
        context.finalize().unwrap();
    }

    #[test]
    fn test_malformed_graphs() {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let graph2 = context.create_graph().unwrap();
        let input1 = graph.input(scalar_type(UINT64)).unwrap();
        let input2 = graph2.input(scalar_type(UINT64)).unwrap();
        let e1 = graph.add(input1.clone(), input2.clone());
        assert!(e1.is_err());
        let fake_node = Node {
            body: Arc::new(AtomicRefCell::new(NodeBody {
                graph: graph.downgrade(),
                node_dependencies: vec![],
                graph_dependencies: vec![],
                operation: Operation::Input(scalar_type(BIT)),
                id: 0,
            })),
        };
        let e2 = graph.add(fake_node.clone(), input1.clone());
        assert!(e2.is_err());
        let fake_node_2 = Node {
            body: Arc::new(AtomicRefCell::new(NodeBody {
                graph: graph.downgrade(),
                node_dependencies: vec![],
                graph_dependencies: vec![],
                operation: Operation::Input(scalar_type(BIT)),
                id: 31337,
            })),
        };
        let e3 = graph.add(fake_node_2.clone(), input1.clone());
        assert!(e3.is_err());
        graph.set_output_node(input1.clone()).unwrap();
        graph.finalize().unwrap();
        let e4 = graph.add(input1.clone(), input1.clone());
        assert!(e4.is_err());
        let graph3 = context.create_graph().unwrap();
        let e5 = graph3.finalize();
        assert!(e5.is_err());
        let e6 = graph3.set_output_node(input1);
        assert!(e6.is_err());
    }

    #[test]
    fn test_malformed_contexts() {
        let context = create_unchecked_context().unwrap();
        let e1 = context.finalize();
        assert!(e1.is_err());
        let graph = context.create_graph().unwrap();
        let e2 = graph.finalize();
        assert!(e2.is_err());
        graph
            .set_output_node(graph.create_tuple(vec![]).unwrap())
            .unwrap();
        let e4 = context.set_main_graph(graph.clone());
        assert!(e4.is_err());
        graph.finalize().unwrap();
        let e3 = context.finalize();
        assert!(e3.is_err());
        context.set_main_graph(graph.clone()).unwrap();
        context.finalize().unwrap();
    }

    #[test]
    fn test_malformed_call_iterate() {
        let context1 = create_unchecked_context().unwrap();
        let graph1 = context1.create_graph().unwrap();
        let output = graph1.create_tuple(vec![]).unwrap();
        graph1.set_output_node(output).unwrap();
        let graph2 = context1.create_graph().unwrap();
        let e1 = graph2.call(graph1.clone(), vec![]);
        assert!(e1.is_err());
        graph1.finalize().unwrap();
        graph2.call(graph1.clone(), vec![]).unwrap();
        let context2 = create_unchecked_context().unwrap();
        let graph3 = context2.create_graph().unwrap();
        let e2 = graph3.call(graph1.clone(), vec![]);
        assert!(e2.is_err());
        let graph4 = context1.create_graph().unwrap();
        graph4.input(tuple_type(vec![])).unwrap();
        graph4.input(tuple_type(vec![])).unwrap();
        let t = graph4.create_tuple(vec![]).unwrap();
        let tt = graph4.create_tuple(vec![t.clone(), t.clone()]).unwrap();
        graph4.set_output_node(tt).unwrap();
        let graph5 = context1.create_graph().unwrap();
        let es = graph5.create_tuple(vec![]).unwrap();
        let v = graph5
            .repeat(graph5.create_tuple(vec![]).unwrap(), 10)
            .unwrap();
        let e3 = graph5.iterate(graph4.clone(), es.clone(), v.clone());
        assert!(e3.is_err());
        graph4.finalize().unwrap();
        graph5
            .iterate(graph4.clone(), es.clone(), v.clone())
            .unwrap();
        let graph6 = context2.create_graph().unwrap();
        let es = graph6.create_tuple(vec![]).unwrap();
        let v = graph6
            .repeat(graph6.create_tuple(vec![]).unwrap(), 10)
            .unwrap();
        let e4 = graph6.iterate(graph4.clone(), es.clone(), v.clone());
        assert!(e4.is_err());
    }

    #[test]
    fn test_graph_consistency() {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let input1 = graph.input(scalar_type(BIT)).unwrap();
        let input2 = graph.input(scalar_type(BIT)).unwrap();
        graph.add(input1.clone(), input2.clone()).unwrap();
        graph.set_output_node(input1.clone()).unwrap();
        graph.finalize().unwrap();
        for (i, node) in graph.get_nodes().iter().enumerate() {
            assert_eq!(node.get_id(), i as u64);
            assert!(graph == node.get_graph());
            for dependency in node.get_node_dependencies() {
                assert!(dependency.get_id() < node.get_id());
            }
        }
        let operations: Vec<Operation> = graph
            .get_nodes()
            .iter()
            .map(|x| x.get_operation())
            .collect();
        assert!(operations.len() == 3);
        if !operations[0].is_input() {
            panic!("Input expected");
        }
        if !operations[1].is_input() {
            panic!("Input expected");
        }
        match operations[2] {
            Operation::Add => {}
            _ => {
                panic!("Add expected");
            }
        }
    }

    #[test]
    fn test_unfinalized_graphs() {
        let context = create_unchecked_context().unwrap();
        let e = context.finalize();
        assert!(e.is_err());
        let graph = context.create_graph().unwrap();
        let graph2 = context.create_graph().unwrap();
        let e = context.finalize();
        assert!(e.is_err());
        let i = graph2.input(scalar_type(BIT)).unwrap();
        graph2.set_output_node(i).unwrap();
        graph2.finalize().unwrap();
        context.set_main_graph(graph2).unwrap();
        let e = context.finalize();
        assert!(e.is_err());
        let ii = graph.input(scalar_type(BIT)).unwrap();
        graph.set_output_node(ii).unwrap();
        graph.finalize().unwrap();
        context.finalize().unwrap();
    }

    #[test]
    fn test_operation_serialization() {
        let o = Operation::Constant(scalar_type(BIT), Value::from_bytes(vec![1]));
        let se = serde_json::to_string(&o).unwrap();
        assert_eq!(
            se,
            format!("{{\"Constant\":[{{\"Scalar\":\"bit\"}},{{\"version\":{},\"data\":\"{{\\\"body\\\":{{\\\"Bytes\\\":[1]}}}}\"}}]}}", DATA_VERSION)
        );
        let de = serde_json::from_str::<Operation>(&se).unwrap();
        assert_eq!(de, o);
    }

    fn context_generators() -> Vec<Box<dyn Fn() -> Context>> {
        let context1 = || {
            let context = create_unchecked_context().unwrap();
            context
        };
        let context2 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            let i = graph.input(scalar_type(BIT)).unwrap();
            graph.set_output_node(i).unwrap();
            graph.finalize().unwrap();
            context.set_main_graph(graph).unwrap();
            context.finalize().unwrap();
            context
        };
        let context3 = || {
            let context = create_unchecked_context().unwrap();
            context.create_graph().unwrap();
            context
        };
        let context4 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            let i = graph.input(scalar_type(BIT)).unwrap();
            graph.set_output_node(i).unwrap();
            graph.finalize().unwrap();
            context
        };
        let context5 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            graph.input(scalar_type(BIT)).unwrap();
            context
        };
        let context6 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            graph
                .constant(scalar_type(BIT), Value::from_bytes(vec![1]))
                .unwrap();
            context
        };
        let context7 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            let i1 = graph.input(scalar_type(BIT)).unwrap();
            let i2 = graph.input(scalar_type(BIT)).unwrap();
            graph.add(i1, i2).unwrap();
            context
        };
        let context8 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            let i1 = graph.input(scalar_type(BIT)).unwrap();
            let i2 = graph.input(scalar_type(BIT)).unwrap();
            graph.add(i2, i1).unwrap();
            context
        };
        let context9 = || {
            let context = create_unchecked_context().unwrap();
            let graph1 = context.create_graph().unwrap();
            let i1 = graph1.input(scalar_type(BIT)).unwrap();
            graph1.set_output_node(i1).unwrap();
            graph1.finalize().unwrap();
            let graph2 = context.create_graph().unwrap();
            let i2 = graph2.input(scalar_type(BIT)).unwrap();
            graph2.set_output_node(i2).unwrap();
            graph2.finalize().unwrap();
            let graph3 = context.create_graph().unwrap();
            let i = graph3.input(scalar_type(BIT)).unwrap();
            graph3.call(graph1, vec![i]).unwrap();
            context
        };
        let context10 = || {
            let context = create_unchecked_context().unwrap();
            let graph1 = context.create_graph().unwrap();
            let i1 = graph1.input(scalar_type(BIT)).unwrap();
            graph1.set_output_node(i1).unwrap();
            graph1.finalize().unwrap();
            let graph2 = context.create_graph().unwrap();
            let i2 = graph2.input(scalar_type(BIT)).unwrap();
            graph2.set_output_node(i2).unwrap();
            graph2.finalize().unwrap();
            let graph3 = context.create_graph().unwrap();
            let i = graph3.input(scalar_type(BIT)).unwrap();
            graph3.call(graph2, vec![i]).unwrap();
            context
        };
        let context11 = || {
            let context = create_unchecked_context().unwrap();
            let graph1 = context.create_graph().unwrap();
            let i1 = graph1.input(scalar_type(BIT)).unwrap();
            graph1.set_output_node(i1).unwrap();
            graph1.finalize().unwrap();
            let graph2 = context.create_graph().unwrap();
            let i2 = graph2.input(scalar_type(BIT)).unwrap();
            graph2.set_output_node(i2).unwrap();
            graph2.finalize().unwrap();
            let graph3 = context.create_graph().unwrap();
            let i = graph3.input(scalar_type(BIT)).unwrap();
            let o = graph3.call(graph2, vec![i]).unwrap();
            graph3.set_output_node(o).unwrap();
            context
        };
        let context12 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            let i = graph.input(scalar_type(BIT)).unwrap();
            graph.set_output_node(i).unwrap();
            graph.finalize().unwrap();
            context.set_main_graph(graph).unwrap();
            context
        };
        let context13 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            let i = graph.input(scalar_type(BIT)).unwrap();
            graph.set_output_node(i).unwrap();
            graph.finalize().unwrap();
            context.set_main_graph(graph.clone()).unwrap();
            context.set_graph_name(graph, "main").unwrap();
            context.finalize().unwrap();
            context
        };
        let context14 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            let i = graph.input(scalar_type(BIT)).unwrap();
            graph.set_output_node(i.clone()).unwrap();
            graph.finalize().unwrap();
            context.set_main_graph(graph.clone()).unwrap();
            context.set_graph_name(graph.clone(), "main").unwrap();
            context.set_node_name(i.clone(), "input").unwrap();
            context
                .add_graph_annotation(&graph, GraphAnnotation::AssociativeOperation)
                .unwrap();
            context
                .add_node_annotation(&i, NodeAnnotation::AssociativeOperation)
                .unwrap();
            context.finalize().unwrap();
            context
        };
        let context15 = || {
            let context = create_unchecked_context().unwrap();
            let graph = context.create_graph().unwrap();
            let mut x = graph.input(scalar_type(BIT)).unwrap();
            for i in 1..20 {
                let y = graph.input(scalar_type(BIT)).unwrap();
                y.set_name(format!("input_{}", i).as_str()).unwrap();
                x = graph.add(x, y).unwrap();
            }
            graph.set_output_node(x).unwrap();
            graph.finalize().unwrap();
            context
        };
        let mut closures: Vec<Box<dyn Fn() -> Context>> = vec![];
        closures.push(Box::new(context1));
        closures.push(Box::new(context2));
        closures.push(Box::new(context3));
        closures.push(Box::new(context4));
        closures.push(Box::new(context5));
        closures.push(Box::new(context6));
        closures.push(Box::new(context7));
        closures.push(Box::new(context8));
        closures.push(Box::new(context9));
        closures.push(Box::new(context10));
        closures.push(Box::new(context11));
        closures.push(Box::new(context12));
        closures.push(Box::new(context13));
        closures.push(Box::new(context14));
        closures.push(Box::new(context15));
        closures
    }

    fn test_context_deep_equal_helper_equal<F>(f: F)
    where
        F: Fn() -> Context,
    {
        let context1 = f();
        let context2 = f();
        assert!(context1 != context2);
        assert!(contexts_deep_equal(context1, context2));
    }

    fn test_context_deep_equal_helper_nonequal<F1, F2>(f1: F1, f2: F2)
    where
        F1: Fn() -> Context,
        F2: Fn() -> Context,
    {
        let context1 = f1();
        let context2 = f2();
        assert!(context1 != context2);
        assert!(!contexts_deep_equal(context1, context2));
    }

    #[test]
    fn test_context_deep_equal() {
        let generators = context_generators();
        for i in 0..generators.len() {
            test_context_deep_equal_helper_equal(&generators[i]);
            for j in 0..i {
                test_context_deep_equal_helper_nonequal(&generators[i], &generators[j]);
            }
        }
    }

    pub fn deserialize_error_lenient(serialized_string: &str, error_msg: &str) {
        use std::panic::catch_unwind;
        panic::set_hook(Box::new(|_info| {
            // See: https://stackoverflow.com/questions/35559267/suppress-panic-output-in-rust-when-using-paniccatch-unwind
        }));
        let result = catch_unwind(|| serde_json::from_str::<Context>(serialized_string).unwrap());
        // This is a (nasty) hack.
        // We check whether the returned error contain the expected error message.
        use ciphercore_utils::execute_main::extract_panic_message;
        if let Err(e) = result {
            match extract_panic_message(e) {
                Some(msg) => {
                    if !msg.contains(error_msg) {
                        panic!("Undesirable panic: {}", msg);
                    }
                }
                None => panic!("Panic of unknown type"),
            }
        } else {
            panic!("Expected error not occur")
        }
    }

    use std::{
        fs::File,
        io::{prelude::*, BufReader},
        path::Path,
    };

    fn lines_from_file(filename: impl AsRef<Path>) -> Vec<String> {
        let file = File::open(filename).expect("no such file");
        let buf = BufReader::new(file);
        buf.lines()
            .map(|l| l.expect("Could not parse line"))
            .collect()
    }

    #[test]
    fn test_context_serialize() {
        let generators = context_generators();
        let contexts: Vec<Context> = generators.iter().map(|generator| generator()).collect();
        let serialized_contexts: Vec<String> = contexts
            .iter()
            .map(|context| serde_json::to_string(context).unwrap())
            .collect();
        let deserialized_contexts: Vec<Context> = serialized_contexts
            .iter()
            .map(|serialized_context| serde_json::from_str(serialized_context).unwrap())
            .collect();
        assert_eq!(contexts.len(), deserialized_contexts.len());
        for i in 0..contexts.len() {
            assert!(contexts[i] != deserialized_contexts[i]);
            assert!(contexts_deep_equal(
                contexts[i].clone(),
                deserialized_contexts[i].clone()
            ));
            assert_eq!(
                serialized_contexts[i],
                serde_json::to_string(&deserialized_contexts[i]).unwrap()
            )
        }

        //Read test cases from golden file
        let test_case = lines_from_file("./src/test_data/version_testcase.txt");
        assert_eq!(serde_json::to_string(&contexts[0]).unwrap(), test_case[0]);

        //Following test case expect an error message "Non-existent main graph" which is caused by the field "\"main_graph\":918276318"
        deserialize_error_lenient(&test_case[1], "Non-existent main graph");
        assert_eq!(serde_json::to_string(&contexts[9]).unwrap(), test_case[2]);
        //Following test case expect an error message "Non-existent node dependency" which is caused by the field "\"node_dependencies\":[918723]"
        deserialize_error_lenient(&test_case[3], "Non-existent node dependency");
        //Following test case expect an error message "Non-existent graph dependency" which is caused by the field "\"graph_dependencies\":[918723]"
        deserialize_error_lenient(&test_case[4], "Non-existent graph dependency");
        assert_eq!(serde_json::to_string(&contexts[13]).unwrap(), test_case[5]);
        //Following test case expect an error message "Non-existent output node" which is caused by the field "\"output_node\":9817273"
        deserialize_error_lenient(&test_case[6], "Non-existent output node");
        //Following test case expect an error message "graphs_names contain an invalid ID" which is caused by the field "\"graphs_names\":[[8079123,\"main\"]]"
        deserialize_error_lenient(&test_case[7], "graphs_names contain an invalid ID");
        //Following test case expect an error message "nodes_names contain an invalid graph ID" which is caused by the field "\"nodes_names\":[[[8079123,0],\"input\"]]"
        deserialize_error_lenient(&test_case[8], "nodes_names contain an invalid graph ID");
        //Following test case expect an error message "nodes_names contain an invalid graph ID" which is caused by the field "\"nodes_names\":[[[0,8079123],\"input\"]]"
        deserialize_error_lenient(&test_case[9], "nodes_names contain an invalid node ID");
        //Following test case expect an error message "Context version doesn't match the requirement" which is caused by its old version number
        deserialize_error_lenient(
            &test_case[10],
            "Context version doesn't match the requirement",
        );
        //Following test case expect an error message "Context version doesn't match the requirement" which is caused by its old version number. Although its payload is unsupported, this should not cause any error before passing the version check.
        deserialize_error_lenient(
            &test_case[11],
            "Context version doesn't match the requirement",
        );
    }

    use crate::data_types::INT32;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use std::iter::FromIterator;

    #[test]
    fn test_named_contexts() {
        let helper = || -> Result<Context> {
            let context = create_context()?;
            let graph = context.create_graph()?;
            let input_a = graph.input(scalar_type(INT32))?;
            let input_b = graph.input(scalar_type(INT32))?;
            let output = graph.add(input_a.clone(), input_b.clone())?;
            graph.set_output_node(output.clone())?;
            graph.finalize()?;
            context.set_main_graph(graph.clone())?;
            assert!(context.get_graph_name(graph.clone()).is_err());
            assert!(context.retrieve_graph("main").is_err());
            assert!(context.get_node_name(input_a.clone())?.is_none());
            assert!(context.retrieve_node(graph.clone(), "a").is_err());
            context.set_graph_name(graph.clone(), "main")?;
            context.set_node_name(input_a.clone(), "a")?;
            assert!(context.retrieve_node(graph.clone(), "b").is_err());
            context.set_node_name(input_b.clone(), "b")?;
            context.finalize()?;
            assert_eq!(context.get_graph_name(graph.clone())?, "main");
            assert_eq!(
                context.get_node_name(input_a.clone())?,
                Some("a".to_owned())
            );
            assert_eq!(
                context.get_node_name(input_b.clone())?,
                Some("b".to_owned())
            );
            assert!(context.retrieve_node(graph.clone(), "a")? == input_a.clone());
            Ok(context)
        };
        let context = helper().unwrap();
        let helper2 = |context: Context| -> Result<i32> {
            let other_context = create_context()?;
            let other_graph = other_context.create_graph()?;
            let input = other_graph.input(scalar_type(BIT))?;
            let other_input = other_graph.input(scalar_type(BIT))?;
            assert!(context
                .prepare_input_values::<Value>(other_graph.clone(), HashMap::new())
                .is_err());
            assert!(other_context
                .prepare_input_values::<Value>(
                    other_graph.clone(),
                    HashMap::from_iter([("a", Value::from_scalar(123, INT32)?)])
                )
                .is_err());
            other_context.set_node_name(input, "b")?;
            assert!(other_context
                .prepare_input_values::<Value>(
                    other_graph.clone(),
                    HashMap::from_iter([("a", Value::from_scalar(123, INT32)?)])
                )
                .is_err());
            assert!(other_context
                .prepare_input_values::<Value>(
                    other_graph.clone(),
                    HashMap::from_iter([("b", Value::from_scalar(123, INT32)?)])
                )
                .is_err());
            other_context.set_node_name(other_input, "c")?;
            assert!(other_context
                .prepare_input_values::<Value>(
                    other_graph,
                    HashMap::from_iter([("b", Value::from_scalar(123, INT32)?)])
                )
                .is_err());
            let g = context.retrieve_graph("main")?;
            let result = random_evaluate(
                g.clone(),
                context.prepare_input_values(
                    g.clone(),
                    HashMap::from_iter([
                        ("a", Value::from_scalar(123, INT32)?),
                        ("b", Value::from_scalar(456, INT32)?),
                    ]),
                )?,
            )?;
            let result = result.to_i32(INT32)?;
            Ok(result)
        };
        assert_eq!(helper2(context).unwrap(), 579);
        let helper3 = |context: Context| -> Result<()> {
            let other_context = create_context()?;
            let other_graph = other_context.create_graph()?;
            let other_node = other_graph.input(scalar_type(BIT))?;
            assert!(context
                .set_graph_name(other_graph.clone(), "outside")
                .is_err());
            assert!(context.get_graph_name(other_graph.clone()).is_err());
            assert!(context
                .set_node_name(other_node.clone(), "outside")
                .is_err());
            assert!(context.get_node_name(other_node.clone()).is_err());
            assert!(context.retrieve_node(other_graph.clone(), "a").is_err());
            Ok(())
        };
        helper3(helper().unwrap()).unwrap();
        let helper4 = || -> Result<()> {
            let context = create_context()?;
            let graph = context.create_graph()?;
            let input = graph.input(scalar_type(BIT))?;
            graph.set_output_node(input.clone())?;
            graph.finalize()?;
            context.set_main_graph(graph.clone())?;
            context.finalize()?;
            assert!(context.set_graph_name(graph, "main").is_err());
            assert!(context.set_node_name(input, "input").is_err());
            Ok(())
        };
        helper4().unwrap();
        let helper5 = || -> Result<()> {
            let context = create_context()?;
            let graph = context.create_graph()?;
            let input = graph.input(scalar_type(BIT))?;
            let other_graph = context.create_graph()?;
            let other_input = graph.input(scalar_type(BIT))?;
            context.set_graph_name(graph.clone(), "main")?;
            assert!(context.set_graph_name(graph, "main3").is_err());
            assert!(context.set_graph_name(other_graph, "main").is_err());
            context.set_node_name(input.clone(), "input")?;
            assert!(context.set_node_name(input, "input3").is_err());
            assert!(context.set_node_name(other_input, "input").is_err());
            Ok(())
        };
        helper5().unwrap();
    }

    #[test]
    fn test_context_type_checking() {
        || -> Result<()> {
            let context = create_context()?;
            let g = context.create_graph()?;
            let i = g.input(tuple_type(vec![]))?;
            assert!(g.add(i.clone(), i.clone()).is_err());
            // Now checking that the node actually have not gotten added by accident
            assert_eq!(g.get_nodes().len(), 1);
            Ok(())
        }()
        .unwrap();
    }

    fn generate_pair_of_equal_contexts() -> Vec<(Context, Context)> {
        let context1 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            let i = g.constant(scalar_type(BIT), Value::from_scalar(0, BIT)?)?;
            g.set_output_node(i)?;
            g.finalize()?;
            g.set_as_main()?;
            Ok(context)
        }()
        .unwrap();
        let context2 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            let i = g.constant(scalar_type(BIT), Value::from_scalar(0, BIT)?)?;
            i.set_as_output()?;
            g.finalize()?;
            context.set_main_graph(g)?;
            Ok(context)
        }()
        .unwrap();
        let context3 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            context.set_graph_name(g, "random graph name")?;
            Ok(context)
        }()
        .unwrap();
        let context4 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            context.create_graph()?.set_name("random graph name")?;
            Ok(context)
        }()
        .unwrap();
        let context5 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            let i = g.constant(scalar_type(BIT), Value::from_scalar(0, BIT)?)?;
            context.set_node_name(i, "random node name")?;
            Ok(context)
        }()
        .unwrap();
        let context6 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            g.constant(scalar_type(BIT), Value::from_scalar(0, BIT)?)?
                .set_name("random node name")?;
            Ok(context)
        }()
        .unwrap();
        let context7 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            let i1 = g.input(scalar_type(BIT))?;
            let i2 = g.input(scalar_type(BIT))?;
            g.add(i1.clone(), i2.clone())?;
            g.subtract(i1.clone(), i2.clone())?;
            g.multiply(i1.clone(), i2.clone())?;
            g.dot(i1.clone(), i2.clone())?;
            g.matmul(i1.clone(), i2.clone())?;
            g.truncate(i1.clone(), 123)?;
            g.sum(i1.clone(), vec![1, 4, 7])?;
            g.permute_axes(i1.clone(), vec![1, 4, 7])?;
            g.get(i1.clone(), vec![1, 4])?;
            g.reshape(i1.clone(), array_type(vec![12, 34], BIT))?;
            g.nop(i1.clone())?;
            g.prf(i1.clone(), 123, scalar_type(BIT))?;
            g.a2b(i1.clone())?;
            g.b2a(i1.clone(), BIT)?;
            g.tuple_get(i1.clone(), 0)?;
            g.named_tuple_get(i1.clone(), "field name".to_owned())?;
            g.vector_get(i1.clone(), i2)?;
            g.array_to_vector(i1.clone())?;
            g.vector_to_array(i1.clone())?;
            g.repeat(i1.clone(), 123)?;
            Ok(context)
        }()
        .unwrap();
        let context8 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            let i1 = g.input(scalar_type(BIT))?;
            let i2 = g.input(scalar_type(BIT))?;
            i1.add(i2.clone())?;
            i1.subtract(i2.clone())?;
            i1.multiply(i2.clone())?;
            i1.dot(i2.clone())?;
            i1.matmul(i2.clone())?;
            i1.truncate(123)?;
            i1.sum(vec![1, 4, 7])?;
            i1.permute_axes(vec![1, 4, 7])?;
            i1.get(vec![1, 4])?;
            i1.reshape(array_type(vec![12, 34], BIT))?;
            i1.nop()?;
            i1.prf(123, scalar_type(BIT))?;
            i1.a2b()?;
            i1.b2a(BIT)?;
            i1.tuple_get(0)?;
            i1.named_tuple_get("field name".to_owned())?;
            i1.vector_get(i2)?;
            i1.array_to_vector()?;
            i1.vector_to_array()?;
            i1.repeat(123)?;
            Ok(context)
        }()
        .unwrap();
        let result = vec![
            (context1, context2),
            (context3, context4),
            (context5, context6),
            (context7, context8),
        ];
        result
    }

    #[test]
    fn test_node_graph_helpers() {
        let pairs_of_contexts = generate_pair_of_equal_contexts();
        for (context1, context2) in pairs_of_contexts {
            assert!(contexts_deep_equal(context1, context2));
        }
        || -> Result<()> {
            let context = create_context()?;
            let g = context.create_graph()?.set_name("graph name")?;
            let i = g.input(scalar_type(BIT))?.set_name("node name")?;
            assert_eq!(g.get_name()?, "graph name");
            assert!(g.retrieve_node("node name")? == i);
            assert_eq!(i.get_name()?, Some("node name".to_owned()));
            assert_eq!(
                g.prepare_input_values(hashmap!("node name" => Value::from_scalar(1, BIT)?))?,
                vec![Value::from_scalar(1, BIT)?]
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_operation_fmt_display() {
        let test_operation_fmt_display_helper = || -> Result<()> {
            let o0 = Rc::new(Operation::Input(scalar_type(UINT16)));
            assert_eq!(format!("{}", o0), "Input");
            let o1 = Rc::new(Operation::Add);
            assert_eq!(format!("{}", o1), "Add");
            let o2 = Rc::new(Operation::Truncate(10));
            assert_eq!(format!("{}", o2), "Truncate");
            let o3 = Rc::new(Operation::Get(vec![10, 20]));
            assert_eq!(format!("{}", o3), "Get");
            let o4 = Rc::new(Operation::NOP);
            assert_eq!(format!("{}", o4), "NOP");
            let o5 = Rc::new(Operation::CreateNamedTuple(vec![
                "Name".to_string(),
                "Address".to_string(),
            ]));
            assert_eq!(format!("{}", o5), "CreateNamedTuple");
            let o6 = Rc::new(Operation::NamedTupleGet("Name".to_string()));
            assert_eq!(format!("{}", o6), "NamedTupleGet");
            Ok(())
        };
        test_operation_fmt_display_helper().unwrap();
    }

    #[test]
    fn test_annotations() {
        let test_annotations_helper = || -> Result<()> {
            let context = create_context()?;
            let g = context.create_graph()?;
            let i = g.input(scalar_type(BIT))?;
            g.add_annotation(GraphAnnotation::AssociativeOperation)?;
            i.add_annotation(NodeAnnotation::AssociativeOperation)?;
            assert_eq!(
                g.get_annotations()?,
                vec![GraphAnnotation::AssociativeOperation]
            );
            assert_eq!(
                i.get_annotations()?,
                vec![NodeAnnotation::AssociativeOperation]
            );
            Ok(())
        };
        test_annotations_helper().unwrap();
    }

    async fn parallel_get_type(output: Node) -> Result<Type> {
        output.get_type()
    }

    async fn parallel_random_evaluate(graph: Graph, context: Context) -> Result<Value> {
        random_evaluate(
            graph.clone(),
            context.prepare_input_values(
                graph,
                HashMap::from_iter([
                    ("one", Value::from_scalar(123, INT32)?),
                    ("two", Value::from_scalar(456, INT32)?),
                ]),
            )?,
        )
    }

    async fn parallel_prepare_for_mpc_evaluation(
        context: Context,
        input_party_map: Vec<Vec<IOStatus>>,
        output_parties: Vec<Vec<IOStatus>>,
        inline_config: InlineConfig,
    ) -> Result<Context> {
        prepare_for_mpc_evaluation(context, input_party_map, output_parties, inline_config)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 50)]
    async fn test_parallel_after_finalize() -> Result<()> {
        let context = create_context()?;
        let graph = context.create_graph()?;
        let input1 = graph.input(scalar_type(INT32))?;
        let input2 = graph.input(scalar_type(INT32))?;
        let output = graph.add(input1.clone(), input2.clone())?;
        graph.set_output_node(output.clone())?;
        graph.finalize()?;
        context.set_main_graph(graph.clone())?;

        context.set_node_name(input1.clone(), "one")?;
        context.set_node_name(input2.clone(), "two")?;

        context.finalize()?;

        assert!(output.clone().get_type().is_ok());

        const PAR_ITERS: usize = 2001;

        let mut get_type_futures = vec![];
        for _ in 0..PAR_ITERS {
            get_type_futures.push(parallel_get_type(output.clone()));
        }
        futures::future::try_join_all(get_type_futures).await?;

        let mut get_random_evaluate_futures = vec![];
        for _ in 0..PAR_ITERS {
            get_random_evaluate_futures
                .push(parallel_random_evaluate(graph.clone(), context.clone()))
        }
        futures::future::try_join_all(get_random_evaluate_futures).await?;

        let input_parties = vec![IOStatus::Party(0), IOStatus::Party(1)];
        let output_parties = vec![IOStatus::Party(0)];

        let mut get_mpc_eval_futures = vec![];
        for _ in 0..PAR_ITERS {
            get_mpc_eval_futures.push(parallel_prepare_for_mpc_evaluation(
                context.clone(),
                vec![input_parties.clone()],
                vec![output_parties.clone()],
                InlineConfig::default(),
            ));
        }
        futures::future::try_join_all(get_mpc_eval_futures).await?;

        Ok(())
    }
}
