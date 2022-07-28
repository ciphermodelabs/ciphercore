use ciphercore_base::custom_ops::CustomOperation;
use ciphercore_base::data_types::{BIT, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8};
use ciphercore_base::typed_value::TypedValue;
use ciphercore_base::{data_types, graphs};
use numpy::PyReadonlyArrayDyn;
use pyo3::class::PyObjectProtocol;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::{
    pyclass, pymethods, pymodule, pyproto, wrap_pyfunction, PyModule, PyResult, Python,
};
use pyo3::PyRef;

macro_rules! wrap {
    ($t:ident, $v:expr) => {
        $t { inner: $v }
    };
}

macro_rules! wrap_vec {
    ($t:ident, $v:expr) => {
        $v.into_iter().map(|x| wrap!($t, x)).collect()
    };
}

macro_rules! unwrap {
    ($v:expr) => {
        $v.inner.clone()
    };
}

macro_rules! unwrap_vec {
    ($v:expr) => {
        $v.into_iter().map(|x| unwrap!(x)).collect()
    };
}

#[pymodule]
fn ciphercore(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    macro_rules! call_serialize_to_str {
        ($n:ident, $t:ident, $v:expr) => {
            #[pyo3::pyfunction]
            fn $n(x: PyReadonlyArrayDyn<$t>) -> PyResult<String> {
                Ok(rust::serialize_to_str(x, $v)?)
            }
            m.add_function(wrap_pyfunction!($n, m)?).unwrap();
        };
    }

    call_serialize_to_str!(serialize_to_str_u64, u64, UINT64);
    call_serialize_to_str!(serialize_to_str_u32, u32, UINT32);
    call_serialize_to_str!(serialize_to_str_u16, u16, UINT16);
    call_serialize_to_str!(serialize_to_str_u8, u8, UINT8);
    call_serialize_to_str!(serialize_to_str_i64, i64, INT64);
    call_serialize_to_str!(serialize_to_str_i32, i32, INT32);
    call_serialize_to_str!(serialize_to_str_i16, i16, INT16);
    call_serialize_to_str!(serialize_to_str_i8, i8, INT8);
    call_serialize_to_str!(serialize_to_str_bit, bool, BIT);

    macro_rules! call_local_shares_and_serialize_to_str {
        ($n:ident, $t:ident, $v:expr) => {
            #[pyo3::pyfunction]
            fn $n(x: PyReadonlyArrayDyn<$t>) -> PyResult<Vec<String>> {
                Ok(rust::local_shares_and_serialize_to_str(x, $v)?)
            }
            m.add_function(wrap_pyfunction!($n, m)?).unwrap();
        };
    }

    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_u64, u64, UINT64);
    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_u32, u32, UINT32);
    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_u16, u16, UINT16);
    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_u8, u8, UINT8);
    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_i64, i64, INT64);
    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_i32, i32, INT32);
    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_i16, i16, INT16);
    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_i8, i8, INT8);
    call_local_shares_and_serialize_to_str!(local_shares_and_serialize_to_str_bit, bool, BIT);

    #[pyfn(m)]
    fn scalar_type<'py>(_py: Python<'py>, st: &ScalarType) -> Type {
        wrap!(Type, data_types::scalar_type(unwrap!(st)))
    }
    #[pyfn(m)]
    fn tuple_type<'py>(_py: Python<'py>, v: Vec<PyRef<Type>>) -> Type {
        wrap!(Type, data_types::tuple_type(unwrap_vec!(v)))
    }
    #[pyfn(m)]
    fn array_type<'py>(_py: Python<'py>, shape: Vec<u64>, st: &ScalarType) -> Type {
        wrap!(Type, data_types::array_type(shape, unwrap!(st)))
    }
    #[pyfn(m)]
    fn vector_type<'py>(_py: Python<'py>, n: u64, t: &Type) -> Type {
        wrap!(Type, data_types::vector_type(n, unwrap!(t)))
    }
    #[pyfn(m)]
    fn named_tuple_type<'py>(_py: Python<'py>, v: Vec<(String, PyRef<Type>)>) -> Type {
        wrap!(
            Type,
            data_types::named_tuple_type(
                v.iter().map(|st| (st.0.clone(), unwrap!(st.1))).collect(),
            )
        )
    }
    #[pyfn(m)]
    fn create_context<'py>(_py: Python<'py>) -> PyResult<Context> {
        Ok(wrap!(Context, graphs::create_context()?))
    }

    m.add("BIT", wrap!(ScalarType, BIT))?;
    m.add("UINT8", wrap!(ScalarType, UINT8))?;
    m.add("INT8", wrap!(ScalarType, INT8))?;
    m.add("UINT16", wrap!(ScalarType, UINT16))?;
    m.add("INT16", wrap!(ScalarType, INT16))?;
    m.add("UINT32", wrap!(ScalarType, UINT32))?;
    m.add("INT32", wrap!(ScalarType, INT32))?;
    m.add("UINT64", wrap!(ScalarType, UINT64))?;
    m.add("INT64", wrap!(ScalarType, INT64))?;
    m.add_class::<ScalarType>()?;
    m.add_class::<Type>()?;
    m.add_class::<Context>()?;
    m.add_class::<Graph>()?;
    m.add_class::<Node>()?;
    m.add_class::<StrTypedValue>()?;
    m.add_class::<StrCustomOperation>()?;
    m.add_class::<SliceElement>()?;
    Ok(())
}

#[pyclass]
struct ScalarType {
    inner: data_types::ScalarType,
}

#[pymethods]
impl ScalarType {
    pub fn size_in_bits(&self) -> u64 {
        data_types::scalar_size_in_bits(unwrap!(self))
    }
}

#[pyclass]
struct Type {
    inner: data_types::Type,
}

#[pymethods]
impl Type {
    pub fn is_scalar(&self) -> bool {
        self.inner.is_scalar()
    }
    pub fn is_array(&self) -> bool {
        self.inner.is_array()
    }
    pub fn is_vector(&self) -> bool {
        self.inner.is_vector()
    }
    pub fn is_tuple(&self) -> bool {
        self.inner.is_tuple()
    }
    pub fn is_named_tuple(&self) -> bool {
        self.inner.is_named_tuple()
    }
    pub fn get_scalar_type(&self) -> ScalarType {
        wrap!(ScalarType, self.inner.get_scalar_type())
    }
    pub fn get_shape(&self) -> Vec<u64> {
        self.inner.get_shape()
    }
    pub fn get_dimensions(&self) -> Vec<u64> {
        self.inner.get_dimensions()
    }
    pub fn size_in_bits(&self) -> PyResult<u64> {
        Ok(data_types::get_size_in_bits(unwrap!(self))?)
    }
}

#[pyclass]
struct Context {
    inner: graphs::Context,
}

#[pymethods]
impl Context {
    pub fn finalize(&self) -> PyResult<Self> {
        Ok(wrap!(Context, self.inner.finalize()?))
    }
    pub fn check_finalized(&self) -> PyResult<()> {
        Ok(self.inner.check_finalized()?)
    }
    pub fn get_num_graphs(&self) -> u64 {
        self.inner.get_num_graphs()
    }
    pub fn contexts_deep_equal(&self, other: &Context) -> bool {
        graphs::contexts_deep_equal(unwrap!(self), unwrap!(other))
    }
    pub fn create_graph(&self) -> PyResult<Graph> {
        Ok(wrap!(Graph, self.inner.create_graph()?))
    }
    pub fn set_main_graph(&self, graph: &Graph) -> PyResult<Context> {
        Ok(wrap!(Context, self.inner.set_main_graph(unwrap!(graph))?))
    }
    pub fn get_graphs(&self) -> Vec<Graph> {
        self.inner
            .get_graphs()
            .iter()
            .map(|g| wrap!(Graph, g.clone()))
            .collect()
    }
    pub fn get_main_graph(&self) -> PyResult<Graph> {
        Ok(wrap!(Graph, self.inner.get_main_graph()?))
    }
    pub fn get_graph_by_id(&self, id: u64) -> PyResult<Graph> {
        Ok(wrap!(Graph, self.inner.get_graph_by_id(id)?))
    }
    pub fn get_node_by_global_id(&self, id: (u64, u64)) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.get_node_by_global_id(id)?))
    }
    pub fn set_graph_name(&self, graph: &Graph, name: &str) -> PyResult<Context> {
        Ok(wrap!(
            Context,
            self.inner.set_graph_name(unwrap!(graph), name)?
        ))
    }
    pub fn get_graph_name(&self, graph: &Graph) -> PyResult<String> {
        Ok(self.inner.get_graph_name(unwrap!(graph))?)
    }
    pub fn retrieve_graph(&self, name: &str) -> PyResult<Graph> {
        Ok(wrap!(Graph, self.inner.retrieve_graph(name)?))
    }
    pub fn set_node_name(&self, node: &Node, name: &str) -> PyResult<Context> {
        Ok(wrap!(
            Context,
            self.inner.set_node_name(unwrap!(node), name)?
        ))
    }
    pub fn get_node_name(&self, node: &Node) -> PyResult<String> {
        Ok(self.inner.get_node_name(unwrap!(node))?)
    }
    pub fn retrieve_node(&self, graph: &Graph, name: &str) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.retrieve_node(unwrap!(graph), name)?))
    }
}

#[pyclass]
struct Graph {
    inner: graphs::Graph,
}

#[pymethods]
impl Graph {
    pub fn input(&self, input_type: &Type) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.input(unwrap!(input_type))?))
    }
    pub fn add(&self, a: &Node, b: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.add(unwrap!(a), unwrap!(b))?))
    }
    pub fn subtract(&self, a: &Node, b: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.subtract(unwrap!(a), unwrap!(b))?))
    }
    pub fn multiply(&self, a: &Node, b: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.multiply(unwrap!(a), unwrap!(b))?))
    }
    pub fn dot(&self, a: &Node, b: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.dot(unwrap!(a), unwrap!(b))?))
    }
    pub fn matmul(&self, a: &Node, b: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.matmul(unwrap!(a), unwrap!(b))?))
    }
    pub fn truncate(&self, a: &Node, b: u64) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.truncate(unwrap!(a), b)?))
    }
    pub fn sum(&self, a: &Node, axes: Vec<u64>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.sum(unwrap!(a), axes)?))
    }
    pub fn permute_axes(&self, a: &Node, axes: Vec<u64>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.permute_axes(unwrap!(a), axes)?))
    }
    pub fn get(&self, a: &Node, axes: Vec<u64>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.get(unwrap!(a), axes)?))
    }
    pub fn get_slice(&self, a: &Node, slices: Vec<PyRef<SliceElement>>) -> PyResult<Node> {
        Ok(wrap!(
            Node,
            self.inner.get_slice(unwrap!(a), unwrap_vec!(slices))?
        ))
    }
    pub fn reshape(&self, a: &Node, t: &Type) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.reshape(unwrap!(a), unwrap!(t))?))
    }
    pub fn random(&self, t: &Type) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.random(unwrap!(t))?))
    }
    pub fn stack(&self, nodes: Vec<PyRef<Node>>, outer_shape: Vec<u64>) -> PyResult<Node> {
        Ok(wrap!(
            Node,
            self.inner.stack(unwrap_vec!(nodes), outer_shape)?
        ))
    }
    pub fn constant(&self, val: &StrTypedValue) -> PyResult<Node> {
        let tv = serde_json::from_str::<TypedValue>(&val.0)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(wrap!(Node, self.inner.constant(tv.t, tv.value)?))
    }
    pub fn a2b(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.a2b(unwrap!(a))?))
    }
    pub fn b2a(&self, a: &Node, st: &ScalarType) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.b2a(unwrap!(a), unwrap!(st))?))
    }
    pub fn create_tuple(&self, elements: Vec<PyRef<Node>>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.create_tuple(unwrap_vec!(elements))?))
    }
    pub fn create_vector(&self, t: &Type, elements: Vec<PyRef<Node>>) -> PyResult<Node> {
        Ok(wrap!(
            Node,
            self.inner
                .create_vector(unwrap!(t), unwrap_vec!(elements))?
        ))
    }
    pub fn create_named_tuple(
        &self,
        elements_nodes: Vec<PyRef<Node>>,
        elements_names: Vec<String>,
    ) -> PyResult<Node> {
        Ok(wrap!(
            Node,
            self.inner.create_named_tuple(
                elements_names
                    .into_iter()
                    .zip(elements_nodes.into_iter().map(|x| unwrap!(x)))
                    .collect()
            )?
        ))
    }
    pub fn tuple_get(&self, a: &Node, ind: u64) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.tuple_get(unwrap!(a), ind)?))
    }
    pub fn named_tuple_get(&self, a: &Node, key: String) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.named_tuple_get(unwrap!(a), key)?))
    }
    pub fn vector_get(&self, a: &Node, index: &Node) -> PyResult<Node> {
        Ok(wrap!(
            Node,
            self.inner.vector_get(unwrap!(a), unwrap!(index))?
        ))
    }
    pub fn zip(&self, elements: Vec<PyRef<Node>>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.zip(unwrap_vec!(elements))?))
    }
    pub fn repeat(&self, a: &Node, n: u64) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.repeat(unwrap!(a), n)?))
    }
    pub fn call(&self, graph: &Graph, arguments: Vec<PyRef<Node>>) -> PyResult<Node> {
        Ok(wrap!(
            Node,
            self.inner.call(unwrap!(graph), unwrap_vec!(arguments))?
        ))
    }
    pub fn iterate(&self, graph: &Graph, state: &Node, input: &Node) -> PyResult<Node> {
        Ok(wrap!(
            Node,
            self.inner
                .iterate(unwrap!(graph), unwrap!(state), unwrap!(input))?
        ))
    }
    pub fn vector_to_array(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.vector_to_array(unwrap!(a))?))
    }
    pub fn array_to_vector(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.array_to_vector(unwrap!(a))?))
    }
    pub fn custom_op(
        &self,
        op: &StrCustomOperation,
        arguments: Vec<PyRef<Node>>,
    ) -> PyResult<Node> {
        let custom_op = serde_json::from_str::<CustomOperation>(&op.0)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(wrap!(
            Node,
            self.inner.custom_op(custom_op, unwrap_vec!(arguments))?
        ))
    }
    pub fn finalize(&self) -> PyResult<Self> {
        Ok(wrap!(Graph, self.inner.finalize()?))
    }
    pub fn get_nodes(&self) -> PyResult<Vec<Node>> {
        Ok(wrap_vec!(Node, self.inner.get_nodes()))
    }
    pub fn set_output_node(&self, a: &Node) -> PyResult<()> {
        Ok(self.inner.set_output_node(unwrap!(a))?)
    }
    pub fn get_output_node(&self) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.get_output_node()?))
    }
    pub fn get_id(&self) -> u64 {
        self.inner.get_id()
    }
    pub fn get_num_nodes(&self) -> u64 {
        self.inner.get_num_nodes()
    }
    pub fn get_node_by_id(&self, id: u64) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.get_node_by_id(id)?))
    }
    pub fn get_context(&self) -> Context {
        wrap!(Context, self.inner.get_context())
    }
    pub fn set_as_main(&self) -> PyResult<Graph> {
        Ok(wrap!(Graph, self.inner.set_as_main()?))
    }
    pub fn set_name(&self, name: String) -> PyResult<Graph> {
        Ok(wrap!(Graph, self.inner.set_name(&name)?))
    }
    pub fn get_name(&self) -> PyResult<String> {
        Ok(self.inner.get_name()?)
    }
    pub fn retrieve_node(&self, name: String) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.retrieve_node(&name)?))
    }
}

#[pyclass(name = "CustomOperation")]
struct StrCustomOperation(String);

#[pymethods]
impl StrCustomOperation {
    #[new]
    fn new(value: String) -> Self {
        StrCustomOperation(value)
    }
}

#[pyclass(name = "TypedValue")]
struct StrTypedValue(String);

#[pymethods]
impl StrTypedValue {
    #[new]
    fn new(value: String) -> Self {
        StrTypedValue(value)
    }
}

#[pyclass]
struct SliceElement {
    inner: graphs::SliceElement,
}

#[pymethods]
impl SliceElement {
    #[staticmethod]
    pub fn from_single_element(ind: i64) -> Self {
        SliceElement {
            inner: graphs::SliceElement::SingleIndex(ind),
        }
    }
    #[staticmethod]
    pub fn from_sub_array(start: Option<i64>, end: Option<i64>, step: Option<i64>) -> Self {
        SliceElement {
            inner: graphs::SliceElement::SubArray(start, end, step),
        }
    }
    #[staticmethod]
    pub fn from_ellipsis() -> Self {
        SliceElement {
            inner: graphs::SliceElement::Ellipsis,
        }
    }
}

#[pyclass]
struct Node {
    inner: graphs::Node,
}

#[pymethods]
impl Node {
    pub fn get_graph(&self) -> Graph {
        wrap!(Graph, self.inner.get_graph())
    }
    pub fn get_dependencies(&self) -> Vec<Node> {
        wrap_vec!(Node, self.inner.get_node_dependencies())
    }
    pub fn get_graph_dependencies(&self) -> Vec<Graph> {
        wrap_vec!(Graph, self.inner.get_graph_dependencies())
    }
    pub fn get_operation(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self.inner.get_operation())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?)
    }
    pub fn get_id(&self) -> u64 {
        self.inner.get_id()
    }
    pub fn get_global_id(&self) -> (u64, u64) {
        self.inner.get_global_id()
    }
    pub fn get_type(&self) -> PyResult<Type> {
        Ok(wrap!(Type, self.inner.get_type()?))
    }
    pub fn add(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.add(unwrap!(a))?))
    }
    pub fn __add__(&self, a: &Node) -> PyResult<Node> {
        self.add(a)
    }
    pub fn sub(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.subtract(unwrap!(a))?))
    }
    pub fn __sub__(&self, a: &Node) -> PyResult<Node> {
        self.sub(a)
    }
    pub fn multiply(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.multiply(unwrap!(a))?))
    }
    pub fn __mul__(&self, a: &Node) -> PyResult<Node> {
        self.multiply(a)
    }
    pub fn matmul(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.matmul(unwrap!(a))?))
    }
    pub fn __matmul__(&self, a: &Node) -> PyResult<Node> {
        self.matmul(a)
    }
    pub fn __and__(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.multiply(unwrap!(a))?))
    }
    pub fn __xor__(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.add(unwrap!(a))?))
    }
    pub fn mixed_multiply(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.mixed_multiply(unwrap!(a))?))
    }
    pub fn dot(&self, a: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.dot(unwrap!(a))?))
    }
    pub fn truncate(&self, b: u64) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.truncate(b)?))
    }
    pub fn sum(&self, axes: Vec<u64>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.sum(axes)?))
    }
    pub fn permute_axes(&self, axes: Vec<u64>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.permute_axes(axes)?))
    }
    pub fn get(&self, axes: Vec<u64>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.get(axes)?))
    }
    pub fn get_slice(&self, slices: Vec<PyRef<SliceElement>>) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.get_slice(unwrap_vec!(slices))?))
    }
    pub fn reshape(&self, t: &Type) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.reshape(unwrap!(t))?))
    }
    pub fn nop(&self) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.nop()?))
    }
    pub fn prf(&self, iv: u64, t: &Type) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.prf(iv, unwrap!(t))?))
    }
    pub fn a2b(&self) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.a2b()?))
    }
    pub fn b2a(&self, st: &ScalarType) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.b2a(unwrap!(st))?))
    }
    pub fn tuple_get(&self, ind: u64) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.tuple_get(ind)?))
    }
    pub fn named_tuple_get(&self, key: String) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.named_tuple_get(key)?))
    }
    pub fn vector_get(&self, index: &Node) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.vector_get(unwrap!(index))?))
    }
    pub fn vector_to_array(&self) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.vector_to_array()?))
    }
    pub fn array_to_vector(&self) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.array_to_vector()?))
    }
    pub fn repeat(&self, n: u64) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.repeat(n)?))
    }
    pub fn set_as_output(&self) -> PyResult<Node> {
        Ok(wrap!(Node, self.inner.set_as_output()?))
    }
}

#[pyproto]
impl PyObjectProtocol for ScalarType {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for Type {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for Context {
    fn __str__(&self) -> PyResult<String> {
        match serde_json::to_string(&self.inner) {
            Ok(s) => Ok(s),
            Err(err) => Err(pyo3::exceptions::PyRuntimeError::new_err(err.to_string())),
        }
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for Graph {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("Graph[num_nodes={}]", self.get_num_nodes()))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for Node {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("Node[type={}]", self.inner.get_type()?))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

mod rust {
    use std::ops::Not;

    use ciphercore_base::data_types::ScalarType;
    use ciphercore_base::errors::Result;
    use ciphercore_base::random::PRNG;
    use ciphercore_base::typed_value::TypedValue;
    use ciphercore_base::typed_value_operations::TypedValueArrayOperations;
    use numpy::PyReadonlyArrayDyn;

    pub(crate) fn serialize_to_str<
        T: numpy::Element + TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy,
    >(
        x: PyReadonlyArrayDyn<T>,
        st: ScalarType,
    ) -> Result<String> {
        let array = x.as_array();
        let tv = TypedValue::from_ndarray(array.to_owned(), st)?;
        Ok(serde_json::to_string(&tv)?)
    }

    pub(crate) fn local_shares_and_serialize_to_str<
        T: numpy::Element + TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy,
    >(
        x: PyReadonlyArrayDyn<T>,
        st: ScalarType,
    ) -> Result<Vec<String>> {
        let array = x.as_array();
        let tv = TypedValue::from_ndarray(array.to_owned(), st)?;
        let mut prng = PRNG::new(None)?;
        let shares = tv.get_local_shares_for_each_party(&mut prng)?;
        let mut result = vec![];
        for share in shares {
            result.push(serde_json::to_string(&share)?)
        }
        Ok(result)
    }
}
