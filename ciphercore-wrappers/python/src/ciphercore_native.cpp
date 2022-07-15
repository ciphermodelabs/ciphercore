#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace CCipherCore {
extern "C" {
#include "ciphercore_adapters.h"
}
} // namespace CCipherCore

namespace PyCipherCore {

[[noreturn]] void handle_error(const CCipherCore::CiphercoreError &err) {
  auto msg = std::string(reinterpret_cast<const char *>(err.msg.ptr));
  cstr_destroy(err.msg);
  throw std::runtime_error(msg.c_str());
}

CCipherCore::CVecVal_u64 vu64_to_CVecVal_u64(std::vector<uint64_t> &data) {
  CCipherCore::CVecVal_u64 result;
  result.ptr = data.data();
  result.len = data.size();
  return result;
}

CCipherCore::CStr s_to_CStr(std::string &data) {
  CCipherCore::CStr result;
  result.ptr = reinterpret_cast<const uint8_t *>(data.c_str());
  return result;
}
std::string CStr_to_string(CCipherCore::CStr s) {
  std::string result(reinterpret_cast<const char *>(s.ptr));
  cstr_destroy(s);
  return result;
}

struct ScalarTypeRaw {
  ScalarTypeRaw(const CCipherCore::ScalarType *st, bool special = false)
      : st(const_cast<CCipherCore::ScalarType *>(st)), special(special) {}
  ~ScalarTypeRaw() {
    if (!special)
      scalar_type_destroy(st);
  }
  CCipherCore::ScalarType *st;
  bool special;
};

template <typename T> auto extract_ok(const T &result) {
  if (result.tag == CCipherCore::Ok) {
    return result.ok;
  }
  handle_error(result.err);
}

struct ScalarType {
  ScalarType(const CCipherCore::ScalarType *st, bool special = false)
      : body(std::make_shared<ScalarTypeRaw>(st, special)) {}

  std::string to_string() const {
    auto result = CCipherCore::scalar_type_to_string(body->st);
    return CStr_to_string(extract_ok(result));
  }

  uint64_t size_in_bits() const {
    auto result = CCipherCore::scalar_size_in_bits(body->st);
    return extract_ok(result);
  }

  std::shared_ptr<ScalarTypeRaw> body;
};

static ScalarType BIT(&CCipherCore::BIT, true);
static ScalarType INT8(&CCipherCore::INT8, true);
static ScalarType UINT8(&CCipherCore::UINT8, true);
static ScalarType INT16(&CCipherCore::INT16, true);
static ScalarType UINT16(&CCipherCore::UINT16, true);
static ScalarType INT32(&CCipherCore::INT32, true);
static ScalarType UINT32(&CCipherCore::UINT32, true);
static ScalarType INT64(&CCipherCore::INT64, true);
static ScalarType UINT64(&CCipherCore::UINT64, true);

struct TypeRaw {
  explicit TypeRaw(CCipherCore::Type *t) : t(t) {}
  ~TypeRaw() { type_destroy(t); }
  CCipherCore::Type *t;
};

struct Type {
  explicit Type(CCipherCore::Type *t) : body(std::make_shared<TypeRaw>(t)) {}

  std::string to_string() const {
    auto result = CCipherCore::type_to_string(body->t);
    return CStr_to_string(extract_ok(result));
  }

  std::shared_ptr<TypeRaw> body;
};

// The motivation for this and other `_Safe` structures is the following.
// We'd like to expose a pointer to some data that is passed to C/Rust.
// At the same time we want the structure to own a corresponding piece of data
// so that it's safe to pass/return to/from functions.
// In spirit, it's similar to `unique_ptr<>`.
struct CVec_Type_Safe {

  CVec_Type_Safe(std::vector<Type> &data) {
    for (const auto &x : data) {
      body.push_back(x.body->t);
    }
  }

  CCipherCore::CVec_Type get() {
    CCipherCore::CVec_Type result;
    result.ptr = body.data();
    result.len = body.size();
    return result;
  }

  std::vector<CCipherCore::Type *> body;
};

struct CVec_u64_Safe {

  CVec_u64_Safe(std::pair<uint64_t, uint64_t> &data) {
    body.push_back(data.first);
    body.push_back(data.second);
  }

  CVec_u64_Safe(std::vector<uint64_t> &data) : body(data) {}

  CCipherCore::CVecVal_u64 get() {
    CCipherCore::CVecVal_u64 result;
    result.ptr = body.data();
    result.len = body.size();
    return result;
  }

  std::vector<uint64_t> body;
};

struct CVecVal_CStr_Safe {

  CVecVal_CStr_Safe(std::vector<std::string> &data) {
    for (auto &x : data) {
      body.push_back(s_to_CStr(x));
    }
  }

  CCipherCore::CVecVal_CStr get() {
    CCipherCore::CVecVal_CStr result;
    result.ptr = body.data();
    result.len = body.size();
    return result;
  }

  std::vector<CCipherCore::CStr> body;
};

Type scalar_type(ScalarType st) {
  auto result = CCipherCore::scalar_type(st.body->st);
  return Type(extract_ok(result));
}

Type array_type(std::vector<uint64_t> shape, ScalarType st) {
  auto result =
      CCipherCore::array_type(vu64_to_CVecVal_u64(shape), st.body->st);
  return Type(extract_ok(result));
}

Type vector_type(uint64_t len, Type t) {
  auto result = CCipherCore::vector_type(len, t.body->t);
  return Type(extract_ok(result));
}

Type tuple_type(std::vector<Type> elements) {
  CVec_Type_Safe cvts(elements);
  auto result = CCipherCore::tuple_type(cvts.get());
  return Type(extract_ok(result));
}

Type named_tuple_type(std::vector<std::pair<std::string, Type>> elements) {
  std::vector<std::string> elements_names;
  std::vector<Type> elements_types;
  for (auto x : elements) {
    elements_names.push_back(x.first);
    elements_types.push_back(x.second);
  }
  CVecVal_CStr_Safe cvcss(elements_names);
  CVec_Type_Safe cvts(elements_types);
  auto result = CCipherCore::named_tuple_type(cvcss.get(), cvts.get());
  return Type(extract_ok(result));
}

struct ContextRaw {
  explicit ContextRaw(CCipherCore::Context *c) : c(c) {}
  ~ContextRaw() { context_destroy(c); }
  CCipherCore::Context *c;
};

struct Graph;
struct Node;

struct Context {
  Context(CCipherCore::Context *c) : body(std::make_shared<ContextRaw>(c)) {}

  Graph create_graph() const;

  Context finalize() const {
    auto result = CCipherCore::context_finalize(body->c);
    return Context(extract_ok(result));
  }

  Context set_main_graph(Graph g) const;

  std::vector<Graph> get_graphs() const;

  bool check_finalized() const {
    auto result = CCipherCore::context_check_finalized(body->c);
    return extract_ok(result);
  }

  Graph get_main_graph() const;

  uint64_t get_num_graphs() const {
    auto result = CCipherCore::context_get_num_graphs(body->c);
    return extract_ok(result);
  }

  Graph get_graph_by_id(uint64_t id) const;

  Node get_node_by_global_id(std::pair<uint64_t, uint64_t> global_id) const;

  std::string to_string() const {
    auto result = CCipherCore::context_to_string(body->c);
    return CStr_to_string(extract_ok(result));
  }

  bool deep_equal(Context other) const {
    auto result = CCipherCore::contexts_deep_equal(body->c, other.body->c);
    return extract_ok(result);
  }

  Context set_graph_name(Graph g, std::string &name) const;

  std::string get_graph_name(Graph g) const;

  Graph retrieve_graph(std::string &name) const;

  Context set_node_name(Node n, std::string &name) const;

  std::string get_node_name(Node n) const;

  Node retrieve_node(Graph g, std::string &name) const;

  std::shared_ptr<ContextRaw> body;
};

struct GraphRaw {
  GraphRaw(CCipherCore::Graph *g) : g(g) {}
  ~GraphRaw() { graph_destroy(g); }
  CCipherCore::Graph *g;
};

// Below are two structures (`MaybeInt64` and `SliceElement`)
// that serve as a proxy between Python slicing and internal slicing definitions
// in the Rust CipherCore. `MaybeInt64` corresponds to `Option<i64>`.

struct MaybeInt64 {

  MaybeInt64() : defined(false), value(0) {}
  MaybeInt64(bool defined, int64_t value) : defined(defined), value(value) {}

  bool defined;
  int64_t value;
};

struct SliceElement {
  enum class Kind {
    SingleIndex,
    SubArray,
    Ellipsis,
  };

  SliceElement() {}

  SliceElement(Kind kind, MaybeInt64 start, MaybeInt64 stop, MaybeInt64 step)
      : kind(kind), start(start), stop(stop), step(step) {}

  Kind kind;
  // For single index, use start.value (and ignore start.defined)
  MaybeInt64 start, stop, step;
};

CCipherCore::CSliceElement
convert_slice_element(const SliceElement &slice_element) {
  CCipherCore::CSliceElement result;
  if (slice_element.kind == SliceElement::Kind::SingleIndex) {
    result.tag = CCipherCore::SingleIndex;
    result.single_index = slice_element.start.value;
  } else if (slice_element.kind == SliceElement::Kind::SubArray) {
    result.tag = CCipherCore::SubArray;
    result.sub_array.op1.valid = slice_element.start.defined;
    result.sub_array.op1.num = slice_element.start.value;
    result.sub_array.op2.valid = slice_element.stop.defined;
    result.sub_array.op2.num = slice_element.stop.value;
    result.sub_array.op3.valid = slice_element.step.defined;
    result.sub_array.op3.num = slice_element.step.value;
  } else {
    result.tag = CCipherCore::Ellipsis;
  }
  return result;
}

SliceElement
convert_slice_element_back(const CCipherCore::CSliceElement &slice_element) {
  SliceElement result;
  if (slice_element.tag == CCipherCore::SingleIndex) {
    result.kind = SliceElement::Kind::SingleIndex;
    result.start.value = slice_element.single_index;
  } else if (slice_element.tag == CCipherCore::SubArray) {
    result.kind = SliceElement::Kind::SubArray;
    result.start.defined = slice_element.sub_array.op1.valid;
    result.start.value = slice_element.sub_array.op1.num;
    result.stop.defined = slice_element.sub_array.op2.valid;
    result.stop.value = slice_element.sub_array.op2.num;
    result.step.defined = slice_element.sub_array.op3.valid;
    result.step.value = slice_element.sub_array.op3.num;
  } else {
    result.kind = SliceElement::Kind::Ellipsis;
  }
  return result;
}

std::vector<SliceElement> CSlice_to_vs(CCipherCore::CSlice *slice) {
  std::vector<SliceElement> result;
  for (size_t i = 0; i < slice->elements.len; ++i) {
    result.push_back(convert_slice_element_back(*slice->elements.ptr[i]));
  }
  c_slice_destroy(slice);
  return result;
}

struct CSlice_Safe {
  CSlice_Safe(std::vector<SliceElement> &slice) : data(slice) {
    for (auto &x : data) {
      data1.push_back(convert_slice_element(x));
    }
    for (auto &x : data1) {
      data2.push_back(&x);
    }
    elements.ptr = data2.data();
    elements.len = data2.size();
    result.elements = elements;
  }

  CCipherCore::CSlice get() { return result; }

  std::vector<SliceElement> data;
  std::vector<CCipherCore::CSliceElement> data1;
  std::vector<CCipherCore::CSliceElement *> data2;
  CCipherCore::CVec_CSliceElement elements;
  CCipherCore::CSlice result;
};

struct Graph {
  Graph(CCipherCore::Graph *g, Context parent_context)
      : body(std::make_shared<GraphRaw>(g)), parent_context(parent_context) {}

  Node input(Type type) const;

  Node add(Node a, Node b) const;

  Node subtract(Node a, Node b) const;

  Node multiply(Node a, Node b) const;

  Node mixed_multiply(Node a, Node b) const;

  Node dot(Node a, Node b) const;

  Node matmul(Node a, Node b) const;

  Node truncate(Node a, uint64_t scale) const;

  Node sum(Node a, std::vector<uint64_t> &axes) const;

  Node permute_axes(Node a, std::vector<uint64_t> &axes) const;

  Node get(Node a, std::vector<uint64_t> &index) const;

  Node get_slice(Node a, std::vector<SliceElement> &slice) const;

  Node reshape(Node a, Type type) const;

  Node random(Type type) const;

  Node stack(std::vector<Node> &nodes,
             std::vector<uint64_t> &outer_shape) const;

  Node constant(std::string &typed_value) const;

  Node a2b(Node a) const;

  Node b2a(Node a, ScalarType st) const;

  Node create_tuple(std::vector<Node> &elements) const;

  Node create_vector(Type type, std::vector<Node> &elements) const;

  Node
  create_named_tuple(std::vector<std::pair<std::string, Node>> &elements) const;

  Node tuple_get(Node a, uint64_t index) const;

  Node named_tuple_get(Node a, std::string &key) const;

  Node vector_get(Node a, Node index) const;

  Node zip(std::vector<Node> &elements) const;

  Node repeat(Node a, uint64_t n) const;

  Node call(Graph callee, std::vector<Node> &arguments) const;

  Node iterate(Graph callee, Node state, Node input) const;

  Node vector_to_array(Node a) const;

  Node array_to_vector(Node a) const;

  Node custom_op(std::string &custom_op, std::vector<Node> &args) const;

  Graph finalize() const;

  std::vector<Node> get_nodes() const;

  bool set_output_node(Node a) const;

  Node get_output_node() const;

  uint64_t get_id() const;

  uint64_t get_num_nodes() const;

  Node get_node_by_id(uint64_t id) const;

  Context get_context() const;

  Graph set_as_main() const;

  Graph set_name(std::string &name) const;

  std::string get_name() const;

  Node retrieve_node(std::string &name) const;

  std::shared_ptr<GraphRaw> body;

  Context parent_context;
};

struct NodeRaw {
  explicit NodeRaw(CCipherCore::Node *n) : n(n) {}
  ~NodeRaw() { node_destroy(n); }
  CCipherCore::Node *n;
};

struct Node {
  Node(CCipherCore::Node *n, Graph parent_graph)
      : body(std::make_shared<NodeRaw>(n)), parent_graph(parent_graph) {}

  Graph get_graph() const;

  std::vector<Node> get_dependencies() const;

  std::vector<Graph> get_graph_dependencies() const;

  pybind11::object get_operation() const;

  uint64_t get_id() const;

  std::pair<uint64_t, uint64_t> get_global_id() const;

  Type get_type() const;

  Node add(Node b) const;

  Node subtract(Node b) const;

  Node multiply(Node b) const;

  Node mixed_multiply(Node b) const;

  Node dot(Node b) const;

  Node matmul(Node b) const;

  Node truncate(uint64_t scale) const;

  Node sum(std::vector<uint64_t> &axes) const;

  Node permute_axes(std::vector<uint64_t> &axes) const;

  Node get(std::vector<uint64_t> &index) const;

  Node get_slice(std::vector<SliceElement> &slice) const;

  Node reshape(Type type) const;

  Node nop() const;

  Node prf(uint64_t iv, Type output_type) const;

  Node a2b() const;

  Node b2a(ScalarType st) const;

  Node tuple_get(uint64_t index) const;

  Node named_tuple_get(std::string &key) const;

  Node vector_get(Node index) const;

  Node array_to_vector() const;

  Node vector_to_array() const;

  Node repeat(uint64_t n) const;

  Node set_as_output() const;

  std::shared_ptr<NodeRaw> body;

  Graph parent_graph;
};

struct CVec_Node_Safe {

  CVec_Node_Safe(std::vector<Node> &data) : body(data) {
    for (auto &x : body) {
      body1.push_back(x.body->n);
    }
  }

  CCipherCore::CVec_Node get() {
    CCipherCore::CVec_Node result;
    result.ptr = body1.data();
    result.len = body1.size();
    return result;
  }

  std::vector<Node> body;
  std::vector<CCipherCore::Node *> body1;
};

Graph Context::create_graph() const {
  auto result = CCipherCore::context_create_graph(body->c);
  return Graph(extract_ok(result), *this);
}

Context Context::set_main_graph(Graph g) const {
  auto result = CCipherCore::context_set_main_graph(body->c, g.body->g);
  return Context(extract_ok(result));
}

std::vector<Graph> CVec_Graph_to_vg(CCipherCore::CVec_Graph *cvg,
                                    Context parent_context) {
  std::vector<Graph> result;
  for (size_t i = 0; i < cvg->len; ++i) {
    result.emplace_back(cvg->ptr[i], parent_context);
  }
  cvec_graph_destroy(cvg);
  return result;
}

std::vector<Graph> Context::get_graphs() const {
  auto result = CCipherCore::context_get_graphs(body->c);
  return CVec_Graph_to_vg(extract_ok(result), *this);
}

Graph Context::get_main_graph() const {
  auto result = CCipherCore::context_get_main_graph(body->c);
  return Graph(extract_ok(result), *this);
}

Graph Context::get_graph_by_id(uint64_t id) const {
  auto result = CCipherCore::context_get_graph_by_id(body->c, id);
  return Graph(extract_ok(result), *this);
}

Node Context::get_node_by_global_id(
    std::pair<uint64_t, uint64_t> global_id) const {
  CVec_u64_Safe cvu64s(global_id);
  auto result =
      CCipherCore::context_get_node_by_global_id(body->c, cvu64s.get());
  return Node(extract_ok(result), get_graph_by_id(global_id.first));
}

Context Context::set_graph_name(Graph g, std::string &name) const {
  auto result =
      CCipherCore::context_set_graph_name(body->c, g.body->g, s_to_CStr(name));
  return Context(extract_ok(result));
}

std::string Context::get_graph_name(Graph g) const {
  auto result = CCipherCore::context_get_graph_name(body->c, g.body->g);
  return CStr_to_string(extract_ok(result));
}

Graph Context::retrieve_graph(std::string &name) const {
  auto result = CCipherCore::context_retrieve_graph(body->c, s_to_CStr(name));
  return Graph(extract_ok(result), *this);
}

Context Context::set_node_name(Node n, std::string &name) const {
  auto result =
      CCipherCore::context_set_node_name(body->c, n.body->n, s_to_CStr(name));
  return Context(extract_ok(result));
}

std::string Context::get_node_name(Node n) const {
  auto result = CCipherCore::context_get_node_name(body->c, n.body->n);
  return CStr_to_string(extract_ok(result));
}

Node Context::retrieve_node(Graph g, std::string &name) const {
  auto result =
      CCipherCore::context_retrieve_node(body->c, g.body->g, s_to_CStr(name));
  return Node(extract_ok(result), g);
}

Context create_context() {
  auto result = CCipherCore::create_context();
  return Context(extract_ok(result));
}

Node Graph::input(Type type) const {
  auto result = CCipherCore::graph_input(body->g, type.body->t);
  return Node(extract_ok(result), *this);
}

Node Graph::add(Node a, Node b) const {
  auto result = CCipherCore::graph_add(body->g, a.body->n, b.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::subtract(Node a, Node b) const {
  auto result = CCipherCore::graph_subtract(body->g, a.body->n, b.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::multiply(Node a, Node b) const {
  auto result = CCipherCore::graph_multiply(body->g, a.body->n, b.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::mixed_multiply(Node a, Node b) const {
  auto result = CCipherCore::graph_mixed_multiply(body->g, a.body->n, b.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::dot(Node a, Node b) const {
  auto result = CCipherCore::graph_dot(body->g, a.body->n, b.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::matmul(Node a, Node b) const {
  auto result = CCipherCore::graph_matmul(body->g, a.body->n, b.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::truncate(Node a, uint64_t scale) const {
  auto result = CCipherCore::graph_truncate(body->g, a.body->n, scale);
  return Node(extract_ok(result), *this);
}

Node Graph::sum(Node a, std::vector<uint64_t> &axes) const {
  CVec_u64_Safe cvu64s(axes);
  auto result = CCipherCore::graph_sum(body->g, a.body->n, cvu64s.get());
  return Node(extract_ok(result), *this);
}

Node Graph::permute_axes(Node a, std::vector<uint64_t> &axes) const {
  CVec_u64_Safe cvu64s(axes);
  auto result =
      CCipherCore::graph_permute_axes(body->g, a.body->n, cvu64s.get());
  return Node(extract_ok(result), *this);
}

Node Graph::get(Node a, std::vector<uint64_t> &index) const {
  CVec_u64_Safe cvu64s(index);
  auto result = CCipherCore::graph_get(body->g, a.body->n, cvu64s.get());
  return Node(extract_ok(result), *this);
}

Node Graph::get_slice(Node a, std::vector<SliceElement> &slice) const {
  CSlice_Safe css(slice);
  auto result = CCipherCore::graph_get_slice(body->g, a.body->n, css.get());
  return Node(extract_ok(result), *this);
}

Node Graph::reshape(Node a, Type type) const {
  auto result = CCipherCore::graph_reshape(body->g, a.body->n, type.body->t);
  return Node(extract_ok(result), *this);
}

Node Graph::random(Type type) const {
  auto result = CCipherCore::graph_random(body->g, type.body->t);
  return Node(extract_ok(result), *this);
}

Node Graph::stack(std::vector<Node> &nodes,
                  std::vector<uint64_t> &outer_shape) const {
  CVec_Node_Safe cvns(nodes);
  CVec_u64_Safe cvu64s(outer_shape);
  auto result = CCipherCore::graph_stack(body->g, cvns.get(), cvu64s.get());
  return Node(extract_ok(result), *this);
}

Node Graph::constant(std::string &typed_value) const {
  CCipherCore::CTypedValue ctv;
  ctv.json = s_to_CStr(typed_value);
  auto result = CCipherCore::graph_constant(body->g, ctv);
  return Node(extract_ok(result), *this);
}

Node Graph::a2b(Node a) const {
  auto result = CCipherCore::graph_a2b(body->g, a.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::b2a(Node a, ScalarType st) const {
  auto result = CCipherCore::graph_b2a(body->g, a.body->n, st.body->st);
  return Node(extract_ok(result), *this);
}

Node Graph::create_tuple(std::vector<Node> &elements) const {
  CVec_Node_Safe cvns(elements);
  auto result = CCipherCore::graph_create_tuple(body->g, cvns.get());
  return Node(extract_ok(result), *this);
}

Node Graph::create_vector(Type type, std::vector<Node> &elements) const {
  CVec_Node_Safe cvns(elements);
  auto result =
      CCipherCore::graph_create_vector(body->g, type.body->t, cvns.get());
  return Node(extract_ok(result), *this);
}

Node Graph::create_named_tuple(
    std::vector<std::pair<std::string, Node>> &elements) const {
  std::vector<std::string> elements_names;
  std::vector<Node> elements_nodes;
  for (auto &x : elements) {
    elements_names.push_back(x.first);
    elements_nodes.push_back(x.second);
  }
  CVecVal_CStr_Safe cvcss(elements_names);
  CVec_Node_Safe cvns(elements_nodes);
  auto result =
      CCipherCore::graph_create_named_tuple(body->g, cvns.get(), cvcss.get());
  return Node(extract_ok(result), *this);
}

Node Graph::tuple_get(Node a, uint64_t index) const {
  auto result = CCipherCore::graph_tuple_get(body->g, a.body->n, index);
  return Node(extract_ok(result), *this);
}

Node Graph::named_tuple_get(Node a, std::string &key) const {
  auto result =
      CCipherCore::graph_named_tuple_get(body->g, a.body->n, s_to_CStr(key));
  return Node(extract_ok(result), *this);
}

Node Graph::vector_get(Node a, Node index) const {
  auto result =
      CCipherCore::graph_vector_get(body->g, a.body->n, index.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::zip(std::vector<Node> &elements) const {
  CVec_Node_Safe cvns(elements);
  auto result = CCipherCore::graph_zip(body->g, cvns.get());
  return Node(extract_ok(result), *this);
}

Node Graph::repeat(Node a, uint64_t n) const {
  auto result = CCipherCore::graph_repeat(body->g, a.body->n, n);
  return Node(extract_ok(result), *this);
}

Node Graph::call(Graph callee, std::vector<Node> &arguments) const {
  CVec_Node_Safe cvns(arguments);
  auto result = CCipherCore::graph_call(body->g, callee.body->g, cvns.get());
  return Node(extract_ok(result), *this);
}

Node Graph::iterate(Graph callee, Node state, Node input) const {
  auto result = CCipherCore::graph_iterate(body->g, callee.body->g,
                                           state.body->n, input.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::vector_to_array(Node a) const {
  auto result = CCipherCore::graph_vector_to_array(body->g, a.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::array_to_vector(Node a) const {
  auto result = CCipherCore::graph_array_to_vector(body->g, a.body->n);
  return Node(extract_ok(result), *this);
}

Node Graph::custom_op(std::string &custom_op, std::vector<Node> &args) const {
  CVec_Node_Safe cvns(args);
  CCipherCore::CCustomOperation cco;
  cco.json = s_to_CStr(custom_op);
  auto result = CCipherCore::graph_custom_op(body->g, cco, cvns.get());
  return Node(extract_ok(result), *this);
}

Graph Graph::finalize() const {
  auto result = CCipherCore::graph_finalize(body->g);
  return Graph(extract_ok(result), parent_context);
}

std::vector<Node> CVec_Node_to_vn(CCipherCore::CVec_Node *cvn,
                                  Graph parent_graph) {
  std::vector<Node> result;
  for (size_t i = 0; i < cvn->len; ++i) {
    result.emplace_back(cvn->ptr[i], parent_graph);
  }
  cvec_node_destroy(cvn);
  return result;
}

std::vector<Node> Graph::get_nodes() const {
  auto result = CCipherCore::graph_get_nodes(body->g);
  return CVec_Node_to_vn(extract_ok(result), *this);
}

bool Graph::set_output_node(Node a) const {
  auto result = CCipherCore::graph_set_output_node(body->g, a.body->n);
  return extract_ok(result);
}

Node Graph::get_output_node() const {
  auto result = CCipherCore::graph_get_output_node(body->g);
  return Node(extract_ok(result), *this);
}

uint64_t Graph::get_id() const {
  auto result = CCipherCore::graph_get_id(body->g);
  return extract_ok(result);
}

uint64_t Graph::get_num_nodes() const {
  auto result = CCipherCore::graph_get_num_nodes(body->g);
  return extract_ok(result);
}

Node Graph::get_node_by_id(uint64_t id) const {
  auto result = CCipherCore::graph_get_node_by_id(body->g, id);
  return Node(extract_ok(result), *this);
}

Context Graph::get_context() const {
  auto result = CCipherCore::graph_get_context(body->g);
  return Context(extract_ok(result));
}

Graph Graph::set_as_main() const {
  auto result = CCipherCore::graph_set_as_main(body->g);
  return Graph(extract_ok(result), parent_context);
}

Graph Graph::set_name(std::string &name) const {
  auto result = CCipherCore::graph_set_name(body->g, s_to_CStr(name));
  return Graph(extract_ok(result), parent_context);
}

std::string Graph::get_name() const {
  auto result = CCipherCore::graph_get_name(body->g);
  return CStr_to_string(extract_ok(result));
}

Node Graph::retrieve_node(std::string &name) const {
  auto result = CCipherCore::graph_retrieve_node(body->g, s_to_CStr(name));
  return Node(extract_ok(result), *this);
}

Graph Node::get_graph() const {
  auto result = CCipherCore::node_get_graph(body->n);
  return Graph(extract_ok(result), parent_graph.parent_context);
}

std::vector<Node> Node::get_dependencies() const {
  auto result = CCipherCore::node_get_dependencies(body->n);
  return CVec_Node_to_vn(extract_ok(result), parent_graph);
}

std::vector<Graph> Node::get_graph_dependencies() const {
  auto result = CCipherCore::node_graph_dependencies(body->n);
  return CVec_Graph_to_vg(extract_ok(result), parent_graph.parent_context);
}

std::vector<uint64_t> CVecVal_u64_to_vu64(CCipherCore::CVecVal_u64 *data) {
  std::vector<uint64_t> result;
  for (size_t i = 0; i < data->len; ++i) {
    result.push_back(data->ptr[i]);
  }
  cvec_u64_destroy(data);
  return result;
}

std::vector<std::string> CVecVal_CStr_to_vs(CCipherCore::CVecVal_CStr *data) {
  std::vector<std::string> result;
  for (size_t i = 0; i < data->len; ++i) {
    result.push_back(reinterpret_cast<const char *>(data->ptr[i].ptr));
    cstr_destroy(data->ptr[i]);
  }
  cvec_cstr_destroy(data);
  return result;
}

enum class OperationKind {
  Input,
  Add,
  Subtract,
  Multiply,
  MixedMultiply
  Dot,
  Matmul,
  Truncate,
  Sum,
  PermuteAxes,
  Get,
  GetSlice,
  Reshape,
  NOP,
  Random,
  PRF,
  Stack,
  Constant,
  A2B,
  B2A,
  CreateTuple,
  CreateNamedTuple,
  CreateVector,
  TupleGet,
  NamedTupleGet,
  VectorGet,
  Zip,
  Repeat,
  Call,
  Iterate,
  ArrayToVector,
  VectorToArray,
  Custom
};

pybind11::object convert_operation(CCipherCore::COperation *operation) {
  if (operation->tag == CCipherCore::Input) {
    return pybind11::make_tuple(OperationKind::Input, Type(operation->input));
  }
  if (operation->tag == CCipherCore::Add) {
    return pybind11::make_tuple(OperationKind::Add);
  }
  if (operation->tag == CCipherCore::Subtract) {
    return pybind11::make_tuple(OperationKind::Subtract);
  }
  if (operation->tag == CCipherCore::Multiply) {
    return pybind11::make_tuple(OperationKind::Multiply);
  }
  if (operation->tag == CCipherCore::MixedMultiply) {
    return pybind11::make_tuple(OperationKind::MixedMultiply);
  }
  if (operation->tag == CCipherCore::Dot) {
    return pybind11::make_tuple(OperationKind::Dot);
  }
  if (operation->tag == CCipherCore::Matmul) {
    return pybind11::make_tuple(OperationKind::Matmul);
  }
  if (operation->tag == CCipherCore::Truncate) {
    return pybind11::make_tuple(OperationKind::Truncate, operation->truncate);
  }
  if (operation->tag == CCipherCore::Sum) {
    return pybind11::make_tuple(OperationKind::Sum,
                                CVecVal_u64_to_vu64(operation->sum));
  }
  if (operation->tag == CCipherCore::PermuteAxes) {
    return pybind11::make_tuple(OperationKind::PermuteAxes,
                                CVecVal_u64_to_vu64(operation->permute_axes));
  }
  if (operation->tag == CCipherCore::Get) {
    return pybind11::make_tuple(OperationKind::Get,
                                CVecVal_u64_to_vu64(operation->get));
  }
  if (operation->tag == CCipherCore::GetSlice) {
    return pybind11::make_tuple(OperationKind::GetSlice,
                                CSlice_to_vs(operation->get_slice));
  }
  if (operation->tag == CCipherCore::Reshape) {
    return pybind11::make_tuple(OperationKind::Reshape,
                                Type(operation->reshape));
  }
  if (operation->tag == CCipherCore::NOP) {
    return pybind11::make_tuple(OperationKind::NOP);
  }
  if (operation->tag == CCipherCore::Random) {
    return pybind11::make_tuple(OperationKind::Random, Type(operation->random));
  }
  if (operation->tag == CCipherCore::PRF) {
    return pybind11::make_tuple(OperationKind::PRF, operation->prf->iv,
                                Type(operation->prf->type_ptr));
  }
  if (operation->tag == CCipherCore::Stack) {
    return pybind11::make_tuple(OperationKind::Stack,
                                CVecVal_u64_to_vu64(operation->stack));
  }
  if (operation->tag == CCipherCore::Constant) {
    return pybind11::make_tuple(OperationKind::Constant,
                                CStr_to_string(operation->constant->json));
  }
  if (operation->tag == CCipherCore::A2B) {
    return pybind11::make_tuple(OperationKind::A2B);
  }
  if (operation->tag == CCipherCore::B2A) {
    return pybind11::make_tuple(OperationKind::B2A, ScalarType(operation->b2a));
  }
  if (operation->tag == CCipherCore::CreateTuple) {
    return pybind11::make_tuple(OperationKind::CreateTuple);
  }
  if (operation->tag == CCipherCore::CreateNamedTuple) {
    return pybind11::make_tuple(
        OperationKind::CreateNamedTuple,
        CVecVal_CStr_to_vs(operation->create_named_tuple));
  }
  if (operation->tag == CCipherCore::CreateVector) {
    return pybind11::make_tuple(OperationKind::CreateVector);
  }
  if (operation->tag == CCipherCore::TupleGet) {
    return pybind11::make_tuple(OperationKind::TupleGet, operation->tuple_get);
  }
  if (operation->tag == CCipherCore::NamedTupleGet) {
    return pybind11::make_tuple(OperationKind::NamedTupleGet,
                                CStr_to_string(operation->named_tuple_get));
  }
  if (operation->tag == CCipherCore::VectorGet) {
    return pybind11::make_tuple(OperationKind::VectorGet);
  }
  if (operation->tag == CCipherCore::Zip) {
    return pybind11::make_tuple(OperationKind::Zip);
  }
  if (operation->tag == CCipherCore::Repeat) {
    return pybind11::make_tuple(OperationKind::Repeat, operation->repeat);
  }
  if (operation->tag == CCipherCore::Call) {
    return pybind11::make_tuple(OperationKind::Call);
  }
  if (operation->tag == CCipherCore::Iterate) {
    return pybind11::make_tuple(OperationKind::Iterate);
  }
  if (operation->tag == CCipherCore::ArrayToVector) {
    return pybind11::make_tuple(OperationKind::ArrayToVector);
  }
  if (operation->tag == CCipherCore::VectorToArray) {
    return pybind11::make_tuple(OperationKind::VectorToArray);
  }
  if (operation->tag == CCipherCore::Custom) {
    return pybind11::make_tuple(OperationKind::Custom,
                                CStr_to_string(operation->custom.json));
  }
  return pybind11::str("Unknown");
}

pybind11::object Node::get_operation() const {
  auto result = CCipherCore::node_get_operation(body->n);
  pybind11::object ret = convert_operation(extract_ok(result));
  c_operation_destroy(extract_ok(result));
  return ret;
}

uint64_t Node::get_id() const {
  auto result = CCipherCore::node_get_id(body->n);
  return extract_ok(result);
}

std::pair<uint64_t, uint64_t>
CVecVal_u64_to_u64u64(CCipherCore::CVecVal_u64 *data) {
  if (data->len != 2) {
    throw std::runtime_error(
        "Error trying to cast a CVecVal_u64 into a pair of u64's");
  }
  std::pair<uint64_t, uint64_t> result;
  result.first = data->ptr[0];
  result.second = data->ptr[1];
  cvec_u64_destroy(data);
  return result;
}

std::pair<uint64_t, uint64_t> Node::get_global_id() const {
  auto result = CCipherCore::node_get_global_id(body->n);
  return CVecVal_u64_to_u64u64(extract_ok(result));
}

Type Node::get_type() const {
  auto result = CCipherCore::node_get_type(body->n);
  return Type(extract_ok(result));
}

Node Node::add(Node b) const {
  auto result = CCipherCore::node_add(body->n, b.body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::subtract(Node b) const {
  auto result = CCipherCore::node_subtract(body->n, b.body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::multiply(Node b) const {
  auto result = CCipherCore::node_multiply(body->n, b.body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::mixed_multiply(Node b) const {
  auto result = CCipherCore::node_mixed_multiply(body->n, b.body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::dot(Node b) const {
  auto result = CCipherCore::node_dot(body->n, b.body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::matmul(Node b) const {
  auto result = CCipherCore::node_matmul(body->n, b.body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::truncate(uint64_t scale) const {
  auto result = CCipherCore::node_truncate(body->n, scale);
  return Node(extract_ok(result), parent_graph);
}

Node Node::sum(std::vector<uint64_t> &axes) const {
  CVec_u64_Safe cvu64s(axes);
  auto result = CCipherCore::node_sum(body->n, cvu64s.get());
  return Node(extract_ok(result), parent_graph);
}

Node Node::permute_axes(std::vector<uint64_t> &axes) const {
  CVec_u64_Safe cvu64s(axes);
  auto result = CCipherCore::node_permute_axes(body->n, cvu64s.get());
  return Node(extract_ok(result), parent_graph);
}

Node Node::get(std::vector<uint64_t> &index) const {
  CVec_u64_Safe cvu64s(index);
  auto result = CCipherCore::node_get(body->n, cvu64s.get());
  return Node(extract_ok(result), parent_graph);
}

Node Node::get_slice(std::vector<SliceElement> &slice) const {
  CSlice_Safe css(slice);
  auto result = CCipherCore::node_get_slice(body->n, css.get());
  return Node(extract_ok(result), parent_graph);
}

Node Node::reshape(Type type) const {
  auto result = CCipherCore::node_reshape(body->n, type.body->t);
  return Node(extract_ok(result), parent_graph);
}

Node Node::nop() const {
  auto result = CCipherCore::node_nop(body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::prf(uint64_t iv, Type output_type) const {
  auto result = CCipherCore::node_prf(body->n, iv, output_type.body->t);
  return Node(extract_ok(result), parent_graph);
}

Node Node::a2b() const {
  auto result = CCipherCore::node_a2b(body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::b2a(ScalarType st) const {
  auto result = CCipherCore::node_b2a(body->n, st.body->st);
  return Node(extract_ok(result), parent_graph);
}

Node Node::tuple_get(uint64_t index) const {
  auto result = CCipherCore::node_tuple_get(body->n, index);
  return Node(extract_ok(result), parent_graph);
}

Node Node::named_tuple_get(std::string &key) const {
  auto result = CCipherCore::node_named_tuple_get(body->n, s_to_CStr(key));
  return Node(extract_ok(result), parent_graph);
}

Node Node::vector_get(Node index) const {
  auto result = CCipherCore::node_vector_get(body->n, index.body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::array_to_vector() const {
  auto result = CCipherCore::node_array_to_vector(body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::vector_to_array() const {
  auto result = CCipherCore::node_vector_to_array(body->n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::repeat(uint64_t n) const {
  auto result = CCipherCore::node_repeat(body->n, n);
  return Node(extract_ok(result), parent_graph);
}

Node Node::set_as_output() const {
  auto result = CCipherCore::node_set_as_output(body->n);
  return Node(extract_ok(result), parent_graph);
}
} // namespace PyCipherCore

PYBIND11_MODULE(ciphercore_native, m) {
  pybind11::class_<PyCipherCore::ScalarType>(m, "ScalarType")
      .def("to_string", &PyCipherCore::ScalarType::to_string)
      .def("__repr__", &PyCipherCore::ScalarType::to_string)
      .def("size_in_bits", &PyCipherCore::ScalarType::size_in_bits);
  m.attr("BIT") = &PyCipherCore::BIT;
  m.attr("INT8") = &PyCipherCore::INT8;
  m.attr("UINT8") = &PyCipherCore::UINT8;
  m.attr("INT16") = &PyCipherCore::INT16;
  m.attr("UINT16") = &PyCipherCore::UINT16;
  m.attr("INT32") = &PyCipherCore::INT32;
  m.attr("UINT32") = &PyCipherCore::UINT32;
  m.attr("INT64") = &PyCipherCore::INT64;
  m.attr("UINT64") = &PyCipherCore::UINT64;
  pybind11::class_<PyCipherCore::Type>(m, "Type")
      .def("to_string", &PyCipherCore::Type::to_string)
      .def("__repr__", &PyCipherCore::Type::to_string);
  m.def("scalar_type", &PyCipherCore::scalar_type);
  m.def("array_type", &PyCipherCore::array_type);
  m.def("vector_type", &PyCipherCore::vector_type);
  m.def("tuple_type", &PyCipherCore::tuple_type);
  m.def("named_tuple_type", &PyCipherCore::named_tuple_type);
  pybind11::class_<PyCipherCore::Context>(m, "Context")
      .def("create_graph", &PyCipherCore::Context::create_graph)
      .def("finalize", &PyCipherCore::Context::finalize)
      .def("set_main_graph", &PyCipherCore::Context::set_main_graph)
      .def("get_graphs", &PyCipherCore::Context::get_graphs)
      .def("check_finalized", &PyCipherCore::Context::check_finalized)
      .def("get_main_graph", &PyCipherCore::Context::get_main_graph)
      .def("get_num_graphs", &PyCipherCore::Context::get_num_graphs)
      .def("get_graph_by_id", &PyCipherCore::Context::get_graph_by_id)
      .def("get_node_by_global_id",
           &PyCipherCore::Context::get_node_by_global_id)
      .def("to_string", &PyCipherCore::Context::to_string)
      .def("__repr__", &PyCipherCore::Context::to_string)
      .def("deep_equal", &PyCipherCore::Context::deep_equal)
      .def("set_graph_name", &PyCipherCore::Context::set_graph_name)
      .def("get_graph_name", &PyCipherCore::Context::get_graph_name)
      .def("retrieve_graph", &PyCipherCore::Context::retrieve_graph)
      .def("set_node_name", &PyCipherCore::Context::set_node_name)
      .def("get_node_name", &PyCipherCore::Context::get_node_name)
      .def("retrieve_node", &PyCipherCore::Context::retrieve_node);
  m.def("create_context", &PyCipherCore::create_context);
  pybind11::class_<PyCipherCore::Graph>(m, "Graph")
      .def("input", &PyCipherCore::Graph::input)
      .def("add", &PyCipherCore::Graph::add)
      .def("subtract", &PyCipherCore::Graph::subtract)
      .def("multiply", &PyCipherCore::Graph::multiply)
      .def("mixed_multiply", &PyCipherCore::Graph::mixed_multiply)
      .def("dot", &PyCipherCore::Graph::dot)
      .def("matmul", &PyCipherCore::Graph::matmul)
      .def("truncate", &PyCipherCore::Graph::truncate)
      .def("sum", &PyCipherCore::Graph::sum)
      .def("permute_axes", &PyCipherCore::Graph::permute_axes)
      .def("get", &PyCipherCore::Graph::get)
      .def("get_slice", &PyCipherCore::Graph::get_slice)
      .def("reshape", &PyCipherCore::Graph::reshape)
      .def("random", &PyCipherCore::Graph::random)
      .def("stack", &PyCipherCore::Graph::stack)
      .def("constant", &PyCipherCore::Graph::constant)
      .def("a2b", &PyCipherCore::Graph::a2b)
      .def("b2a", &PyCipherCore::Graph::b2a)
      .def("create_tuple", &PyCipherCore::Graph::create_tuple)
      .def("create_vector", &PyCipherCore::Graph::create_vector)
      .def("create_named_tuple", &PyCipherCore::Graph::create_named_tuple)
      .def("tuple_get", &PyCipherCore::Graph::tuple_get)
      .def("named_tuple_get", &PyCipherCore::Graph::named_tuple_get)
      .def("vector_get", &PyCipherCore::Graph::vector_get)
      .def("zip", &PyCipherCore::Graph::zip)
      .def("repeat", &PyCipherCore::Graph::repeat)
      .def("call", &PyCipherCore::Graph::call)
      .def("iterate", &PyCipherCore::Graph::iterate)
      .def("vector_to_array", &PyCipherCore::Graph::vector_to_array)
      .def("array_to_vector", &PyCipherCore::Graph::array_to_vector)
      .def("custom_op", &PyCipherCore::Graph::custom_op)
      .def("finalize", &PyCipherCore::Graph::finalize)
      .def("get_nodes", &PyCipherCore::Graph::get_nodes)
      .def("set_output_node", &PyCipherCore::Graph::set_output_node)
      .def("get_output_node", &PyCipherCore::Graph::get_output_node)
      .def("get_id", &PyCipherCore::Graph::get_id)
      .def("get_num_nodes", &PyCipherCore::Graph::get_num_nodes)
      .def("get_node_by_id", &PyCipherCore::Graph::get_node_by_id)
      .def("get_context", &PyCipherCore::Graph::get_context)
      .def("set_as_main", &PyCipherCore::Graph::set_as_main)
      .def("set_name", &PyCipherCore::Graph::set_name)
      .def("get_name", &PyCipherCore::Graph::get_name)
      .def("retrieve_node", &PyCipherCore::Graph::retrieve_node);
  pybind11::class_<PyCipherCore::Node>(m, "Node")
      .def("get_graph", &PyCipherCore::Node::get_graph)
      .def("get_dependencies", &PyCipherCore::Node::get_dependencies)
      .def("get_graph_dependencies",
           &PyCipherCore::Node::get_graph_dependencies)
      .def("get_operation", &PyCipherCore::Node::get_operation)
      .def("get_id", &PyCipherCore::Node::get_id)
      .def("get_global_id", &PyCipherCore::Node::get_global_id)
      .def("get_type", &PyCipherCore::Node::get_type)
      .def("add", &PyCipherCore::Node::add)
      .def("subtract", &PyCipherCore::Node::subtract)
      .def("multiply", &PyCipherCore::Node::multiply)
      .def("mixed_multiply", &PyCipherCore::Node::mixed_multiply)
      .def("dot", &PyCipherCore::Node::dot)
      .def("matmul", &PyCipherCore::Node::matmul)
      .def("truncate", &PyCipherCore::Node::truncate)
      .def("sum", &PyCipherCore::Node::sum)
      .def("permute_axes", &PyCipherCore::Node::permute_axes)
      .def("get", &PyCipherCore::Node::get)
      .def("get_slice", &PyCipherCore::Node::get_slice)
      .def("reshape", &PyCipherCore::Node::reshape)
      .def("nop", &PyCipherCore::Node::nop)
      .def("prf", &PyCipherCore::Node::prf)
      .def("a2b", &PyCipherCore::Node::a2b)
      .def("b2a", &PyCipherCore::Node::b2a)
      .def("tuple_get", &PyCipherCore::Node::tuple_get)
      .def("named_tuple_get", &PyCipherCore::Node::named_tuple_get)
      .def("vector_get", &PyCipherCore::Node::vector_get)
      .def("array_to_vector", &PyCipherCore::Node::array_to_vector)
      .def("vector_to_array", &PyCipherCore::Node::vector_to_array)
      .def("repeat", &PyCipherCore::Node::repeat)
      .def("set_as_output", &PyCipherCore::Node::set_as_output);

  pybind11::class_<PyCipherCore::MaybeInt64>(m, "MaybeInt64")
      .def(pybind11::init<bool, int64_t>());

  pybind11::class_<PyCipherCore::SliceElement> slice_element(m, "SliceElement");
  slice_element.def(
      pybind11::init<PyCipherCore::SliceElement::Kind, PyCipherCore::MaybeInt64,
                     PyCipherCore::MaybeInt64, PyCipherCore::MaybeInt64>());

  pybind11::enum_<PyCipherCore::SliceElement::Kind>(slice_element, "Kind")
      .value("SingleIndex", PyCipherCore::SliceElement::Kind::SingleIndex)
      .value("SubArray", PyCipherCore::SliceElement::Kind::SubArray)
      .value("Ellipsis", PyCipherCore::SliceElement::Kind::Ellipsis);
  pybind11::enum_<PyCipherCore::OperationKind>(m, "OperationKind")
      .value("Input", PyCipherCore::OperationKind::Input)
      .value("Add", PyCipherCore::OperationKind::Add)
      .value("Subtract", PyCipherCore::OperationKind::Subtract)
      .value("Multiply", PyCipherCore::OperationKind::Multiply)
      .value("MixedMultiply", PyCipherCore::OperationKind::MixedMultiply)
      .value("Dot", PyCipherCore::OperationKind::Dot)
      .value("Matmul", PyCipherCore::OperationKind::Matmul)
      .value("Truncate", PyCipherCore::OperationKind::Truncate)
      .value("Sum", PyCipherCore::OperationKind::Sum)
      .value("PermuteAxes", PyCipherCore::OperationKind::PermuteAxes)
      .value("Get", PyCipherCore::OperationKind::Get)
      .value("GetSlice", PyCipherCore::OperationKind::GetSlice)
      .value("Reshape", PyCipherCore::OperationKind::Reshape)
      .value("NOP", PyCipherCore::OperationKind::NOP)
      .value("Random", PyCipherCore::OperationKind::Random)
      .value("PRF", PyCipherCore::OperationKind::PRF)
      .value("Stack", PyCipherCore::OperationKind::Stack)
      .value("Constant", PyCipherCore::OperationKind::Constant)
      .value("A2B", PyCipherCore::OperationKind::A2B)
      .value("B2A", PyCipherCore::OperationKind::B2A)
      .value("CreateTuple", PyCipherCore::OperationKind::CreateTuple)
      .value("CreateNamedTuple", PyCipherCore::OperationKind::CreateNamedTuple)
      .value("CreateVector", PyCipherCore::OperationKind::CreateVector)
      .value("TupleGet", PyCipherCore::OperationKind::TupleGet)
      .value("NamedTupleGet", PyCipherCore::OperationKind::NamedTupleGet)
      .value("VectorGet", PyCipherCore::OperationKind::VectorGet)
      .value("Zip", PyCipherCore::OperationKind::Zip)
      .value("Repeat", PyCipherCore::OperationKind::Repeat)
      .value("Call", PyCipherCore::OperationKind::Call)
      .value("Iterate", PyCipherCore::OperationKind::Iterate)
      .value("ArrayToVector", PyCipherCore::OperationKind::ArrayToVector)
      .value("VectorToArray", PyCipherCore::OperationKind::VectorToArray)
      .value("Custom", PyCipherCore::OperationKind::Custom);
}
