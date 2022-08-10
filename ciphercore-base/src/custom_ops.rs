//! Structs and traits necessary to implement custom operations.
//! A custom operation can be thought of as a polymorphic function, i.e., where the number of inputs and their types can vary.
//! Two basic examples of custom operations are provided: [Not] and [Or].
use crate::data_types::{scalar_type, Type, BIT};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{copy_node_name, create_context, Context, Graph, Node, Operation};

use serde::{Deserialize, Serialize};

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};

use std::any::{Any, TypeId};
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::fmt::Debug;
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[cfg(feature = "py-binding")]
use pywrapper_macro::struct_wrapper;

#[doc(hidden)]
/// This trait can be used to compare and hash trait objects.
/// Based on
/// <https://stackoverflow.com/questions/25339603/how-to-test-for-equality-between-trait-objects>
/// and
/// <https://stackoverflow.com/questions/64838355/how-do-i-create-a-hashmap-with-type-erased-keys>.
pub trait DynEqHash {
    fn as_any(&self) -> &dyn Any;
    fn equals(&self, _: &dyn Any) -> bool;
    fn hash(&self) -> u64;
}

impl<T: 'static + Eq + Hash> DynEqHash for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<T>().map_or(false, |a| self == a)
    }

    /// To hash an instance of `T`, we hash a pair (identifier of the type T, value)
    /// using the `DefaultHasher` of `hash_map`, which seems to be implementing SipHash
    /// (<https://github.com/veorq/SipHash/>).
    fn hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        Hash::hash(&(TypeId::of::<T>(), self), &mut h);
        h.finish()
    }
}

/// A trait that must be implemented by any custom operation struct.
///
/// Only structures satisfying this trait can be used to create [CustomOperation].
///
/// Any structure implementing this trait must also implement the following traits:
/// - [Debug],
/// - [Serialize],
/// - [Deserialize],
/// - [Eq],
/// - [PartialEq],
/// - [Hash]
///
/// # Example
/// This is the actual implementation of the custom operation [Not].
/// ```
/// use serde::{Deserialize, Serialize};
/// # use ciphercore_base::data_types::{BIT, scalar_type, Type};
/// # use ciphercore_base::data_values::Value;
/// # use ciphercore_base::graphs::{Context, Graph};
/// # use ciphercore_base::custom_ops::{CustomOperationBody};
/// # use ciphercore_base::errors::Result;
/// # use ciphercore_base::runtime_error;
///
/// #[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
/// pub struct Not {}
/// #[typetag::serde] // requires the typetag crate
/// impl CustomOperationBody for Not {
///    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
///        if arguments_types.len() != 1 {
///           return Err(runtime_error!("Invalid number of arguments for Not"));
///        }
///        let g = context.create_graph()?;
///        g.input(arguments_types[0].clone())?
///         .add(g.constant(scalar_type(BIT), Value::from_scalar(1, BIT)?)?)?
///         .set_as_output()?;
///        g.finalize()?;
///        Ok(g)
///    }
///    fn get_name(&self) -> String {
///        "Not".to_owned()
///    }
/// }
/// ```
///
// This is used to add a concrete type tag when serializing
// Any `impl CustomOperationBody` should have
// #[typetag::serde] before it
#[typetag::serde(tag = "type")]
pub trait CustomOperationBody: 'static + Debug + DynEqHash + Send + Sync {
    /// Defines the logic of a custom operation.
    ///
    /// This function must create a graph in a given context computing a custom operation.
    /// Note that that the number of inputs and their types can vary.
    /// This function should describe the logic of the custom operation for all acceptable cases and return an error otherwise.
    ///
    /// # Arguments
    ///
    /// * `context` - context where a graph computing a custom operation should be created
    /// * `arguments_types` - vector of input types of a custom operation
    ///
    /// # Returns
    ///
    /// New graph computing a custom operation
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph>;

    /// Specifies and returns the name of this custom operation.
    ///
    /// The name must be unique among all the implemented custom operations.
    ///
    /// # Returns
    ///
    /// Name of this custom operation
    fn get_name(&self) -> String;
}

/// A structure that stores a pointer to a custom operation.
///
/// A custom operation can be thought of as a polymorphic function, i.e., where the number of inputs and their types can vary.
///
/// Any struct can be a custom operation if it implements the [CustomOperationBody] trait.
/// Then any such struct can be used to create a [CustomOperation] object that can be added to a computation graph with [Graph::custom_op].
///
/// # Rust crates
///
/// [Clone] trait duplicates the pointer, not the underlying custom operation.
///
/// [PartialEq] trait compares the related custom operations, not just pointer.
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
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "py-binding", struct_wrapper)]
pub struct CustomOperation {
    body: Arc<dyn CustomOperationBody>,
}

#[cfg(feature = "py-binding")]
#[pyo3::pymethods]
impl PyBindingCustomOperation {
    #[new]
    fn new(value: String) -> pyo3::PyResult<Self> {
        let custom_op = serde_json::from_str::<CustomOperation>(&value)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        Ok(PyBindingCustomOperation { inner: custom_op })
    }
    fn __str__(&self) -> pyo3::PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
    }
    fn __repr__(&self) -> pyo3::PyResult<String> {
        self.__str__()
    }
}

impl CustomOperation {
    /// Creates a new custom operation that can be added to a computation graph via [Graph::custom_op].
    ///
    /// # Arguments
    ///
    /// `op` - struct that implements the [CustomOperationBody] trait
    ///
    /// # Returns
    ///
    /// New custom operation
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
    pub fn new<T: 'static + CustomOperationBody>(op: T) -> CustomOperation {
        CustomOperation { body: Arc::new(op) }
    }

    /// Returns the name of the underlying custom operation by calling [CustomOperationBody::get_name].
    ///
    /// # Returns
    ///
    /// Name of this custom operation
    pub fn get_name(&self) -> String {
        self.body.get_name()
    }
}

impl CustomOperation {
    #[doc(hidden)]
    pub fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        self.body.instantiate(context, arguments_types)
    }
}

impl PartialEq for CustomOperation {
    /// Tests whether `self` and `other` custom operations are equal.
    ///
    /// The underlying custom operation structs are compared via the [Eq] trait.
    ///
    /// # Arguments
    ///
    /// `other` - another [CustomOperation]
    ///
    /// # Returns
    ///
    /// `true` if the pointers in `self` and `other` point to equal custom operations, `false` otherwise
    fn eq(&self, other: &Self) -> bool {
        self.body.equals((*other.body).as_any())
    }
}

impl Hash for CustomOperation {
    /// Hashes the custom operation pointer.
    ///
    /// # Arguments
    ///
    /// `state` - state of a hash function that is changed after hashing the custom operation
    fn hash<H: Hasher>(&self, state: &mut H) {
        let hash_value = DynEqHash::hash(self.body.as_ref());
        state.write_u64(hash_value);
    }
}

impl Eq for CustomOperation {}

/// A structure that defines the custom operation Not that inverts elementwise a binary array or scalar (individual bit).
///
/// This operation accepts only a binary array or scalar as input.
///
/// To use this and other custom operations in computation graphs, see [Graph::custom_op].
///
/// # Custom operation arguments
///
/// Node containing a binary array or scalar
///
/// # Custom operation returns
///
/// New Not node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation, Not};
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = scalar_type(BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.custom_op(CustomOperation::new(Not {}), vec![n1]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Not {}

#[typetag::serde]
impl CustomOperationBody for Not {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!("Invalid number of arguments for Not"));
        }
        let g = context.create_graph()?;
        g.input(arguments_types[0].clone())?
            .add(g.constant(scalar_type(BIT), Value::from_scalar(1, BIT)?)?)?
            .set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "Not".to_owned()
    }
}

/// A structure that defines the custom operation Or that is equivalent to the binary Or applied elementwise.
///
/// This operation accepts only binary arrays or scalars as input.
///
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2,3]`.
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
/// New Or node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation, Or};
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = scalar_type(BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(Or {}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Or {}

#[typetag::serde]
impl CustomOperationBody for Or {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 2 {
            return Err(runtime_error!("Invalid number of arguments for Or"));
        }
        let g = context.create_graph()?;
        let i1 = g.input(arguments_types[0].clone())?;
        let i2 = g.input(arguments_types[1].clone())?;
        let i1_not = g.custom_op(CustomOperation::new(Not {}), vec![i1])?;
        let i2_not = g.custom_op(CustomOperation::new(Not {}), vec![i2])?;
        g.custom_op(CustomOperation::new(Not {}), vec![i1_not.multiply(i2_not)?])?
            .set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "Or".to_owned()
    }
}

#[doc(hidden)]
/// Data structure for storing maps between two contexts.
/// Can be used conveniently to glue one context into another.
/// Note that `Node` and `Graph` are hashed as pointers,
/// so in general it's not a good idea to enumerate the entries
/// of the maps using iterators, since it's not deterministic.
/// Instead, one should sort by IDs, and then enumerate.
#[derive(Default)]
pub struct ContextMappings {
    node_mapping: HashMap<Node, Node>,
    graph_mapping: HashMap<Graph, Graph>,
}

impl ContextMappings {
    pub fn contains_graph(&self, graph: Graph) -> bool {
        self.graph_mapping.contains_key(&graph)
    }

    pub fn contains_node(&self, node: Node) -> bool {
        self.node_mapping.contains_key(&node)
    }

    /// Panics if `graph` is not in `graph_mapping`
    pub fn get_graph(&self, graph: Graph) -> Graph {
        self.graph_mapping
            .get(&graph)
            .expect("Graph is not found in graph_mapping")
            .clone()
    }

    /// Panics if `node` is not in `node_mapping`
    pub fn get_node(&self, node: Node) -> Node {
        self.node_mapping
            .get(&node)
            .expect("Node is not found in node_mapping")
            .clone()
    }

    /// Panics if `old_graph` has already been inserted
    pub fn insert_graph(&mut self, old_graph: Graph, new_graph: Graph) {
        assert!(
            self.graph_mapping.insert(old_graph, new_graph).is_none(),
            "Graph has already been inserted in graph_mapping"
        );
    }

    /// Panics if `old_node` has already been inserted
    pub fn insert_node(&mut self, old_node: Node, new_node: Node) {
        assert!(
            self.node_mapping.insert(old_node, new_node).is_none(),
            "Node has already been inserted in node_mapping"
        );
    }

    /// Panics if `old_graph` is not inserted
    pub fn remove_graph(&mut self, old_graph: Graph) {
        assert!(
            self.graph_mapping.remove(&old_graph).is_some(),
            "Graph is not in graph_mapping"
        );
    }

    /// Panics if `old_node` is not inserted
    pub fn remove_node(&mut self, old_node: Node) {
        assert!(
            self.node_mapping.remove(&old_node).is_some(),
            "Node is not isn node_mapping"
        );
    }
}

#[doc(hidden)]
pub struct MappedContext {
    pub context: Context,
    // old -> new mappings
    pub mappings: ContextMappings,
}

impl MappedContext {
    pub fn new(context: Context) -> Self {
        MappedContext {
            context,
            mappings: ContextMappings::default(),
        }
    }

    pub fn get_context(&self) -> Context {
        self.context.clone()
    }
}

/// An instantiation is given by a custom operation
/// and types of the arguments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct Instantiation {
    pub(super) op: CustomOperation,
    pub(super) arguments_types: Vec<Type>,
}

impl Instantiation {
    /// `create_from_node` assumes that `node` carries a custom operation,
    /// and the ambient context is type checked.
    fn create_from_node(node: Node) -> Result<Self> {
        if let Operation::Custom(custom_op) = node.get_operation() {
            let mut node_dependencies_types = vec![];
            for dependency in node.get_node_dependencies() {
                node_dependencies_types.push(dependency.get_type()?);
            }
            Ok(Instantiation {
                op: custom_op,
                arguments_types: node_dependencies_types,
            })
        } else {
            Err(runtime_error!(
                "Instantiations can only be created from custom nodes"
            ))
        }
    }

    fn get_name(&self) -> String {
        let mut name = "__".to_owned();
        name.push_str(&self.op.get_name());
        name.push_str("::<");
        let mut first_argument = true;
        for t in &self.arguments_types {
            if first_argument {
                first_argument = false;
            } else {
                name.push_str(", ");
            }
            write!(name, "{}", t).unwrap();
        }
        name.push('>');
        name
    }
}

/// Data structures for instantiation graph.
type InstantiationsGraph = DiGraph<Instantiation, (), usize>;
type InstantiationsGraphNode = NodeIndex<usize>;

/// Mapping between the graph of instantiations of actual instantiations.
#[derive(Default)]
struct InstantiationsGraphMapping {
    instantiation_to_node: HashMap<Instantiation, InstantiationsGraphNode>,
    node_to_instantiation: HashMap<InstantiationsGraphNode, Instantiation>,
}

/// Retrieves a node of the instantiations graph
/// (and adds a fresh one if necessary).
fn get_instantiations_graph_node(
    instantiation: &Instantiation,
    instantiations_graph_mapping: &mut InstantiationsGraphMapping,
    instantiations_graph: &mut InstantiationsGraph,
) -> (InstantiationsGraphNode, bool) {
    match instantiations_graph_mapping
        .instantiation_to_node
        .get(instantiation)
    {
        Some(id) => (*id, true),
        None => {
            let new_inode = instantiations_graph.add_node(instantiation.clone());
            instantiations_graph_mapping
                .instantiation_to_node
                .insert(instantiation.clone(), new_inode);
            instantiations_graph_mapping
                .node_to_instantiation
                .insert(new_inode, instantiation.clone());
            (new_inode, false)
        }
    }
}

/// Recursive function to find all the necessary instantiations
/// and build a graph of those (appropriately caching things).
fn process_instantiation(
    instantiation: &Instantiation,
    instantiations_graph_mapping: &mut InstantiationsGraphMapping,
    instantiations_graph: &mut InstantiationsGraph,
) -> Result<()> {
    let fake_context = create_context()?;
    let graph = instantiation
        .op
        .instantiate(fake_context.clone(), instantiation.arguments_types.clone())?;
    // `instantiate()` may potentially create some auxiliary graphs, which we now need to process
    // TODO: add a test that check that this is properly done
    for fake_graph in fake_context.get_graphs() {
        for node in fake_graph.get_nodes() {
            if let Operation::Custom(_) = node.get_operation() {
                let new_instantiation = Instantiation::create_from_node(node)?;
                let (node1, already_existed) = get_instantiations_graph_node(
                    &new_instantiation,
                    instantiations_graph_mapping,
                    instantiations_graph,
                );
                let (node2, _) = get_instantiations_graph_node(
                    instantiation,
                    instantiations_graph_mapping,
                    instantiations_graph,
                );
                instantiations_graph.add_edge(node1, node2, ());
                if !already_existed {
                    process_instantiation(
                        &new_instantiation,
                        instantiations_graph_mapping,
                        instantiations_graph,
                    )?;
                }
            }
        }
    }
    graph.set_as_main()?;
    fake_context.finalize()?;
    Ok(())
}

#[doc(hidden)]
/// In order to instantiate all the custom operations in a given context,
/// we do the following. First, we build a graph of instantiations as follows.
/// As a seed set, we use the instantiated custom operations in the original context.
/// Then, starting from this seed set, we detect custom operations necessary further
/// down the road, and instantiate them recursively etc. If we deal with instantiation
/// we already encountered, we stop.
///
/// After we built the graph, we sort it topologically. Finally, we glue
/// all the necessary instantiations into the resulting context followed by the
/// original context.
///
/// For instance, if we want to instantiate Or from inputs of type
/// `Type::Array(vec![1, 7], BIT)` and `Type::Array(vec![3, 7], BIT)`,
/// then we need Not for types `Type::Array(vec![1, 7], BIT)` once
/// and for `Type::Array(vec![3, 7], BIT)` twice. But the latter will be instantiated
/// only once due to caching.
pub fn run_instantiation_pass(context: Context) -> Result<MappedContext> {
    /* Build a graph of instantiations */
    let mut needed_instantiations = vec![];
    for graph in context.get_graphs() {
        for node in graph.get_nodes() {
            if let Operation::Custom(_) = node.get_operation() {
                needed_instantiations.push(Instantiation::create_from_node(node)?);
            }
        }
    }
    let mut instantiations_graph_mapping = InstantiationsGraphMapping::default();
    let mut instantiations_graph = InstantiationsGraph::default();
    for instantiation in needed_instantiations {
        let (_, already_existed) = get_instantiations_graph_node(
            &instantiation,
            &mut instantiations_graph_mapping,
            &mut instantiations_graph,
        );
        if !already_existed {
            process_instantiation(
                &instantiation,
                &mut instantiations_graph_mapping,
                &mut instantiations_graph,
            )?;
        }
    }
    /* =============================== */
    let result_context = create_context()?;
    // Glues a given context into the final one
    let glue_context = |glued_instantiations_cache: &HashMap<Instantiation, Graph>,
                        context_to_glue: Context|
     -> Result<ContextMappings> {
        let mut mapping = ContextMappings::default();
        for graph_to_glue in context_to_glue.get_graphs() {
            let glued_graph = result_context.create_graph()?;
            for annotation in graph_to_glue.get_annotations()? {
                glued_graph.add_annotation(annotation)?;
            }
            mapping.insert_graph(graph_to_glue.clone(), glued_graph.clone());
            for node in graph_to_glue.get_nodes() {
                let node_dependencies = node.get_node_dependencies();
                let new_node_dependencies: Vec<Node> = node_dependencies
                    .iter()
                    .map(|node| mapping.get_node(node.clone()))
                    .collect();
                let new_node = match node.get_operation() {
                    Operation::Custom(_) => {
                        let needed_instantiation = Instantiation::create_from_node(node.clone())?;
                        glued_graph.call(
                            // Retrieve a needed instantiation from the cache,
                            // which should be glued before.
                            glued_instantiations_cache
                                .get(&needed_instantiation)
                                .expect("Should not be here")
                                .clone(),
                            new_node_dependencies,
                        )?
                    }
                    _ => {
                        let graph_dependencies = node.get_graph_dependencies();
                        let new_graph_dependencies: Vec<Graph> = graph_dependencies
                            .iter()
                            .map(|graph| mapping.get_graph(graph.clone()))
                            .collect();
                        glued_graph.add_node(
                            new_node_dependencies,
                            new_graph_dependencies,
                            node.get_operation(),
                        )?
                    }
                };
                copy_node_name(node.clone(), new_node.clone())?;
                let node_annotations = context_to_glue.get_node_annotations(node.clone())?;
                if !node_annotations.is_empty() {
                    for node_annotation in node_annotations {
                        new_node.add_annotation(node_annotation)?;
                    }
                }
                mapping.insert_node(node, new_node);
            }
            glued_graph.set_output_node(mapping.get_node(graph_to_glue.get_output_node()?))?;
            glued_graph.finalize()?;
        }
        Ok(mapping)
    };
    // Glue necessary instantiations in the order of toposort of
    // the instantiations graph, and add them to the cache.
    let mut glued_instantiations_cache = HashMap::<_, Graph>::new();
    for instantiations_graph_node in toposort(&instantiations_graph, None)
        .map_err(|_| runtime_error!("Circular dependency among instantiations"))?
    {
        let instantiation = instantiations_graph_mapping
            .node_to_instantiation
            .get(&instantiations_graph_node)
            .expect("Should not be here");
        let fake_context = create_context()?;
        let g = instantiation
            .op
            .instantiate(fake_context.clone(), instantiation.arguments_types.clone())?
            .set_as_main()?;
        fake_context.finalize()?;
        let mapping = glue_context(&glued_instantiations_cache, fake_context)?;
        let mapped_graph = mapping.get_graph(g);
        mapped_graph.set_name(&instantiation.get_name())?;
        glued_instantiations_cache.insert(instantiation.clone(), mapped_graph);
    }
    // Glue the final context.
    let mut result = MappedContext::new(result_context.clone());
    result.mappings = glue_context(&glued_instantiations_cache, context.clone())?;
    result_context.set_main_graph(result.mappings.get_graph(context.get_main_graph()?))?;
    result_context.finalize()?;
    Ok(result)
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::data_types::array_type;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::{contexts_deep_equal, NodeAnnotation};

    fn get_hash(custom_op: &CustomOperation) -> u64 {
        let mut h = DefaultHasher::new();
        Hash::hash(custom_op, &mut h);
        h.finish()
    }

    #[test]
    fn test_custom_operation() {
        assert_eq!(CustomOperation::new(Not {}), CustomOperation::new(Not {}));
        assert_eq!(CustomOperation::new(Or {}), CustomOperation::new(Or {}));
        assert!(CustomOperation::new(Not {}) != CustomOperation::new(Or {}));
        assert_eq!(
            get_hash(&CustomOperation::new(Not {})),
            get_hash(&CustomOperation::new(Not {})),
        );
        assert_eq!(
            get_hash(&CustomOperation::new(Or {})),
            get_hash(&CustomOperation::new(Or {})),
        );
        assert!(get_hash(&CustomOperation::new(Or {})) != get_hash(&CustomOperation::new(Not {})),);
        let v = vec![CustomOperation::new(Not {}), CustomOperation::new(Or {})];
        let sers = vec![
            "{\"body\":{\"type\":\"Not\"}}",
            "{\"body\":{\"type\":\"Or\"}}",
        ];
        let debugs = vec![
            "CustomOperation { body: Not }",
            "CustomOperation { body: Or }",
        ];
        for i in 0..v.len() {
            let s = serde_json::to_string(&v[i]).unwrap();
            assert_eq!(s, sers[i]);
            assert_eq!(serde_json::from_str::<CustomOperation>(&s).unwrap(), v[i]);
            assert_eq!(v, v.clone());
            assert_eq!(format!("{:?}", v[i]), debugs[i]);
        }
        assert!(serde_json::from_str::<CustomOperation>(
            "{\"body\":{\"type\":\"InvalidCustomOperation\"}}"
        )
        .is_err());
    }

    #[test]
    fn test_not() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i = g.input(scalar_type(BIT))?;
            let o = g.custom_op(CustomOperation::new(Not {}), vec![i])?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g.clone())?;
            c.finalize()?;
            let mapped_c = run_instantiation_pass(c)?;
            for x in vec![0, 1] {
                let result = random_evaluate(
                    mapped_c.mappings.get_graph(g.clone()),
                    vec![Value::from_scalar(x, BIT)?],
                )?;
                let result = result.to_u8(BIT)?;
                assert_eq!(result, !(x != 0) as u8);
            }
            Ok(())
        }()
        .unwrap();
        // Test broadcasting
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i = g.input(array_type(vec![3, 3], BIT))?;
            let o = g.custom_op(CustomOperation::new(Not {}), vec![i])?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g.clone())?;
            c.finalize()?;
            let mapped_c = run_instantiation_pass(c)?;
            let result = random_evaluate(
                mapped_c.mappings.get_graph(g.clone()),
                vec![Value::from_flattened_array(
                    &vec![0, 1, 1, 0, 1, 0, 0, 1, 1],
                    BIT,
                )?],
            )?;
            let result = result.to_flattened_array_u64(array_type(vec![3, 3], BIT))?;
            assert_eq!(result, vec![1, 0, 0, 1, 0, 1, 1, 0, 0]);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_or() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.input(scalar_type(BIT))?;
            let i2 = g.input(scalar_type(BIT))?;
            let o = g.custom_op(CustomOperation::new(Or {}), vec![i1, i2])?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g.clone())?;
            c.finalize()?;
            let mapped_c = run_instantiation_pass(c)?;
            for x in vec![0, 1] {
                for y in vec![0, 1] {
                    let result = random_evaluate(
                        mapped_c.mappings.get_graph(g.clone()),
                        vec![Value::from_scalar(x, BIT)?, Value::from_scalar(y, BIT)?],
                    )?;
                    let result = result.to_u8(BIT)?;
                    assert_eq!(result, ((x != 0) || (y != 0)) as u8);
                }
            }
            Ok(())
        }()
        .unwrap();
    }

    #[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
    struct A {}

    #[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
    struct B {}

    #[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
    struct C {}

    #[typetag::serde]
    impl CustomOperationBody for A {
        fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
            let g = context.create_graph()?;
            g.custom_op(
                CustomOperation::new(B {}),
                vec![g.input(arguments_types[0].clone())?],
            )?
            .set_as_output()?;
            g.finalize()?;
            Ok(g)
        }

        fn get_name(&self) -> String {
            "A".to_owned()
        }
    }

    #[typetag::serde]
    impl CustomOperationBody for B {
        fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
            let g = context.create_graph()?;
            let i = g.input(arguments_types[0].clone())?;
            g.set_output_node(i)?;
            g.finalize()?;
            let fake_g = context.create_graph()?;
            let i = fake_g.input(scalar_type(BIT))?;
            fake_g.set_output_node(i)?;
            fake_g.finalize()?;
            Ok(g)
        }

        fn get_name(&self) -> String {
            "B".to_owned()
        }
    }

    #[typetag::serde]
    impl CustomOperationBody for C {
        fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
            let g = context.create_graph()?;
            let mut inputs = vec![];
            for t in &arguments_types {
                inputs.push(g.input(t.clone())?);
            }
            let o = if arguments_types.len() == 1 {
                inputs[0].clone()
            } else {
                let node = g.create_tuple(vec![
                    g.custom_op(
                        CustomOperation::new(C {}),
                        inputs[0..inputs.len() / 2].to_vec(),
                    )?,
                    g.custom_op(
                        CustomOperation::new(C {}),
                        inputs[inputs.len() / 2..inputs.len()].to_vec(),
                    )?,
                ])?;
                context.add_node_annotation(&node, NodeAnnotation::AssociativeOperation)?;
                node
            };
            g.set_output_node(o)?;
            g.finalize()?;
            Ok(g)
        }

        fn get_name(&self) -> String {
            "C".to_owned()
        }
    }

    #[test]
    fn test_instantiation_pass() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i = g.input(scalar_type(BIT))?;
            let o = g.custom_op(CustomOperation::new(A {}), vec![i])?;
            o.set_name("A")?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g)?;
            c.finalize()?;

            let processed_c = run_instantiation_pass(c)?.context;

            let expected_c = create_context()?;
            let g2 = expected_c.create_graph()?;
            let i = g2.input(scalar_type(BIT))?;
            g2.set_output_node(i)?;
            g2.set_name("__B::<b>")?;
            g2.finalize()?;
            let g1 = expected_c.create_graph()?;
            let i = g1.input(scalar_type(BIT))?;
            g1.set_output_node(i)?;
            g1.finalize()?;
            let g3 = expected_c.create_graph()?;
            let i = g3.input(scalar_type(BIT))?;
            let o = g3.call(g2, vec![i])?;
            g3.set_output_node(o)?;
            g3.set_name("__A::<b>")?;
            g3.finalize()?;
            let g4 = expected_c.create_graph()?;
            let i = g4.input(scalar_type(BIT))?;
            let o = g4.call(g3, vec![i])?;
            o.set_name("A")?;
            g4.set_output_node(o)?;
            g4.finalize()?;
            expected_c.set_main_graph(g4)?;
            expected_c.finalize()?;
            assert!(contexts_deep_equal(expected_c, processed_c));
            Ok(())
        }()
        .unwrap();

        || -> Result<()> {
            let c = create_context()?;
            let sub_g = c.create_graph()?;
            let i = sub_g.input(scalar_type(BIT))?;
            sub_g.set_output_node(i)?;
            sub_g.finalize()?;
            let g = c.create_graph()?;
            let i = g.input(scalar_type(BIT))?;
            let ii = g.call(sub_g, vec![i])?;
            let o = g.custom_op(CustomOperation::new(B {}), vec![ii])?;
            o.set_name("B")?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g)?;
            c.finalize()?;

            let processed_c = run_instantiation_pass(c)?.context;

            let expected_c = create_context()?;
            let g1 = expected_c.create_graph()?;
            let i = g1.input(scalar_type(BIT))?;
            g1.set_output_node(i)?;
            g1.set_name("__B::<b>")?;
            g1.finalize()?;
            let g3 = expected_c.create_graph()?;
            let i = g3.input(scalar_type(BIT))?;
            g3.set_output_node(i)?;
            g3.finalize()?;
            let g2 = expected_c.create_graph()?;
            let i = g2.input(scalar_type(BIT))?;
            g2.set_output_node(i)?;
            g2.finalize()?;
            let g4 = expected_c.create_graph()?;
            let i = g4.input(scalar_type(BIT))?;
            let o = g4.call(g2, vec![i])?;
            let oo = g4.call(g1, vec![o])?;
            oo.set_name("B")?;
            g4.set_output_node(oo)?;
            g4.finalize()?;
            expected_c.set_main_graph(g4)?;
            expected_c.finalize()?;
            assert!(contexts_deep_equal(expected_c, processed_c));
            Ok(())
        }()
        .unwrap();

        // Checking that `run_instantiation_pass` is deterministic
        || -> Result<()> {
            let generate_context = || -> Result<Context> {
                let c = create_context()?;
                let g = c.create_graph()?;
                let i1 = g.input(array_type(vec![1, 5], BIT))?;
                let i2 = g.input(array_type(vec![7, 5], BIT))?;
                let i3 = g.input(array_type(vec![4, 3], BIT))?;
                let i4 = g.input(array_type(vec![2, 3], BIT))?;
                let o = g.custom_op(CustomOperation::new(C {}), vec![i1, i2, i3, i4])?;
                g.set_output_node(o)?;
                g.finalize()?;
                c.set_main_graph(g)?;
                c.finalize()?;
                Ok(c)
            };
            let mut contexts = vec![];
            for _ in 0..10 {
                contexts.push(generate_context()?);
            }
            let mut instantiated_contexts = vec![];
            for context in contexts {
                instantiated_contexts.push(run_instantiation_pass(context)?.context);
            }
            for i in 0..instantiated_contexts.len() {
                assert!(contexts_deep_equal(
                    instantiated_contexts[0].clone(),
                    instantiated_contexts[i].clone()
                ));
            }
            Ok(())
        }()
        .unwrap();

        // Checking that `run_instantiation_pass` copies node annotations
        || -> Result<()> {
            let generate_context = || -> Result<Context> {
                let c = create_context()?;
                let g = c.create_graph()?;
                let i1 = g.input(array_type(vec![1, 5], BIT))?;
                let i2 = g.input(array_type(vec![7, 5], BIT))?;
                let i3 = g.input(array_type(vec![4, 3], BIT))?;
                let i4 = g.input(array_type(vec![2, 3], BIT))?;
                let o = g.custom_op(CustomOperation::new(C {}), vec![i1, i2, i3, i4])?;
                g.set_output_node(o)?;
                g.finalize()?;
                c.set_main_graph(g)?;
                c.finalize()?;
                Ok(c)
            };
            let new_context = run_instantiation_pass(generate_context()?)?.context;
            assert_eq!(
                new_context
                    .get_node_annotations(new_context.get_graphs()[6].get_output_node()?)?
                    .len(),
                1
            );
            Ok(())
        }()
        .unwrap();

        // Check `run_instantiation_pass` for Not
        || -> Result<()> {
            let generate_context = || -> Result<Context> {
                let c = create_context()?;
                let g = c.create_graph()?;
                let i1 = g.input(array_type(vec![5], BIT))?;
                let o = g.custom_op(CustomOperation::new(Not {}), vec![i1])?;
                g.set_output_node(o)?;
                g.finalize()?;
                c.set_main_graph(g)?;
                c.finalize()?;
                Ok(c)
            };
            let c = generate_context()?;
            let mapped_c = run_instantiation_pass(c)?;
            let expected_c = create_context()?;
            let not_g = expected_c.create_graph()?;
            let i = not_g.input(array_type(vec![5], BIT))?;
            let c = not_g.constant(scalar_type(BIT), Value::from_bytes(vec![1]))?;
            let o = not_g.add(i, c)?;
            not_g.set_output_node(o)?;
            not_g.set_name("__Not::<b[5]>")?;
            not_g.finalize()?;
            let g = expected_c.create_graph()?;
            let i = g.input(array_type(vec![5], BIT))?;
            let o = g.call(not_g, vec![i])?;
            g.set_output_node(o)?;
            g.finalize()?;
            expected_c.set_main_graph(g)?;
            expected_c.finalize()?;
            assert!(contexts_deep_equal(mapped_c.context, expected_c));
            Ok(())
        }()
        .unwrap();

        // Check `run_instantiation_pass` for Or
        || -> Result<()> {
            let generate_context = || -> Result<Context> {
                let c = create_context()?;
                let g = c.create_graph()?;
                let i1 = g.input(array_type(vec![5], BIT))?;
                let i2 = g.input(array_type(vec![3, 5], BIT))?;
                let o = g.custom_op(CustomOperation::new(Or {}), vec![i1, i2])?;
                g.set_output_node(o)?;
                g.finalize()?;
                c.set_main_graph(g)?;
                c.finalize()?;
                Ok(c)
            };
            let c = generate_context()?;
            let mapped_c = run_instantiation_pass(c)?;
            let expected_c = create_context()?;
            let not_g_2 = expected_c.create_graph()?;
            let i = not_g_2.input(array_type(vec![3, 5], BIT))?;
            let c = not_g_2.constant(scalar_type(BIT), Value::from_bytes(vec![1]))?;
            let o = not_g_2.add(i, c)?;
            not_g_2.set_output_node(o)?;
            not_g_2.set_name("__Not::<b[3, 5]>")?;
            not_g_2.finalize()?;
            let not_g = expected_c.create_graph()?;
            let i = not_g.input(array_type(vec![5], BIT))?;
            let c = not_g.constant(scalar_type(BIT), Value::from_bytes(vec![1]))?;
            let o = not_g.add(i, c)?;
            not_g.set_output_node(o)?;
            not_g.set_name("__Not::<b[5]>")?;
            not_g.finalize()?;
            let or_g = expected_c.create_graph()?;
            let i1 = or_g.input(array_type(vec![5], BIT))?;
            let i2 = or_g.input(array_type(vec![3, 5], BIT))?;
            let i1_not = or_g.call(not_g, vec![i1])?;
            let i2_not = or_g.call(not_g_2.clone(), vec![i2])?;
            let i1_not_and_i2_not = or_g.multiply(i1_not, i2_not)?;
            let o = or_g.call(not_g_2, vec![i1_not_and_i2_not])?;
            or_g.set_output_node(o)?;
            or_g.set_name("__Or::<b[5], b[3, 5]>")?;
            or_g.finalize()?;
            let g = expected_c.create_graph()?;
            let i1 = g.input(array_type(vec![5], BIT))?;
            let i2 = g.input(array_type(vec![3, 5], BIT))?;
            let o = g.call(or_g, vec![i1, i2])?;
            g.set_output_node(o)?;
            g.finalize()?;
            expected_c.set_main_graph(g)?;
            expected_c.finalize()?;
            assert!(contexts_deep_equal(mapped_c.context, expected_c));
            Ok(())
        }()
        .unwrap();
    }
}
