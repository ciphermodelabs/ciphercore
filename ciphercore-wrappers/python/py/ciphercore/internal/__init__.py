import ciphercore_internal as cc
from string import Template

BIT = cc.BIT
INT8 = cc.INT8
INT16 = cc.INT16
INT32 = cc.INT32
INT64 = cc.INT64
UINT8 = cc.UINT8
UINT16 = cc.UINT16
UINT32 = cc.UINT32
UINT64 = cc.UINT64

# Re-export type-related primitives.
Value = cc.Value
Type = cc.Type
ScalarType = cc.ScalarType
array_type = cc.array_type
scalar_type = cc.scalar_type
tuple_type = cc.tuple_type
vector_type = cc.vector_type
named_tuple_type = cc.named_tuple_type

# Re-export graph-related primitives.

_MAGIC_METHOD_ALLOWLIST = ['__repr__', '__str__']

# We wrap Context/Graph/Node in proxy objects so that we can store pointers from Graph to Context, and from Node to Graph.


class Context(object):
    def __init__(self, inner):
        self._inner = inner


def _downcast_tuple(args, cls):
    res = []
    for arg in args:
        if isinstance(arg, cls):
            res.append(arg._inner)
        else:
            res.append(arg)
    return tuple(res)


def _downcast_list(args, cls):
    res = []
    for arg in args:
        if isinstance(arg, list):
            res.append(_downcast_list(arg, cls))
        elif isinstance(arg, tuple):
            res.append(_downcast_tuple(arg, cls))
        elif isinstance(arg, cls):
            res.append(arg._inner)
        else:
            res.append(arg)
    return res


def _downcast_dict(kwargs, cls):
    res = {}
    for k, arg in kwargs.items():
        if isinstance(arg, list):
            res[k] = _downcast_list(arg, cls)
        elif isinstance(arg, cls):
            res[k] = arg._inner
        else:
            res[k] = arg
    return res


def _wrap_result(res, ctx):
    if isinstance(res, list):
        return [_wrap_result(x, ctx) for x in res]
    if isinstance(res, cc.Node):
        return Node(res, ctx)
    if isinstance(res, cc.Graph):
        return Graph(res, ctx)
    if isinstance(res, cc.Context):
        return Context(res)
    return res


def get_context_wrapper(name):
    def wrapper(self, *args, **kwargs):
        args = _downcast_list(args, Node)
        kwargs = _downcast_dict(kwargs, Node)
        args = _downcast_list(args, Graph)
        kwargs = _downcast_dict(kwargs, Graph)
        args = _downcast_list(args, Context)
        kwargs = _downcast_dict(kwargs, Context)
        return _wrap_result(getattr(self._inner, name)(*args, **kwargs), self)
    return wrapper


def wrap_class(cc_cls, cls, wrapper_fn):
    for name in dir(cc_cls):
        if not callable(getattr(cc_cls, name)):
            continue
        if name.startswith('__') and name not in _MAGIC_METHOD_ALLOWLIST:
            continue
        setattr(cls, name, wrapper_fn(name))
        getattr(cls, name).__doc__ = getattr(cc_cls, name).__doc__
        getattr(cls, name).__text_signature__ = getattr(
            cc_cls, name).__text_signature__


wrap_class(cc.Context, Context, get_context_wrapper)


def create_context():
    return Context(cc.create_context())


create_context.__doc__ = cc.create_context.__doc__
create_context.__text_signature__ = cc.create_context.__text_signature__


def get_graph_wrapper(name):
    def wrapper(self, *args, **kwargs):
        args = _downcast_list(args, Node)
        kwargs = _downcast_dict(kwargs, Node)
        args = _downcast_list(args, Graph)
        kwargs = _downcast_dict(kwargs, Graph)
        return _wrap_result(getattr(self._inner, name)(*args, **kwargs), self.context)

    return wrapper


class Graph(object):
    def __init__(self, inner, context):
        self._inner = inner
        self.context = context


wrap_class(cc.Graph, Graph, get_graph_wrapper)


def get_node_wrapper(name):
    def wrapper(self, *args, **kwargs):
        args = _downcast_list(args, Node)
        kwargs = _downcast_dict(kwargs, Node)
        return _wrap_result(getattr(self._inner, name)(*args, **kwargs), self.context)

    return wrapper


CustomOperation = cc.CustomOperation


def get_signed(node):
    return node.get_type().get_scalar_type().get_signed()


def get_signed_value(a, b):
    return 'true' if get_signed(a) or get_signed(b) else 'false'


_comparison_op_template = Template(
    '{"body":{"type":"$type","signed_comparison":$signed}}')


class Node(object):
    def __init__(self, inner, context):
        self._inner = inner
        self.context = context

    def _comparison_op_helper(self, other, op_name):
        g = self.get_graph()
        op = CustomOperation(_comparison_op_template.substitute(
            type=op_name, signed=get_signed_value(self, other)))
        return g.custom_op(op, [self, other])

    def __lt__(self, other):
        name = 'LessThan'
        return self._comparison_op_helper(other, name)

    def __gt__(self, other):
        name = 'GreaterThan'
        return self._comparison_op_helper(other, name)

    def __le__(self, other):
        name = 'LessThanEqualTo'
        return self._comparison_op_helper(other, name)

    def __ge__(self, other):
        name = 'GreaterThanEqualTo'
        return self._comparison_op_helper(other, name)

    def __eq__(self, other):
        name = 'Equal'
        return self._comparison_op_helper(other, name)

    def __ne__(self, other):
        name = 'Equal'
        return self._comparison_op_helper(other, name)

    def min(self, other):
        name = 'Min'
        return self._comparison_op_helper(other, name)

    def max(self, other):
        name = 'Max'
        return self._comparison_op_helper(other, name)


wrap_class(cc.Node, Node, get_node_wrapper)

# Syntactic sugar: overriding operators.
Node.__add__ = Node.add
Node.__sub__ = Node.subtract
Node.__mul__ = Node.multiply
Node.__matmul__ = Node.matmul
Node.__and__ = Node.multiply
Node.__xor__ = Node.add

# Add constructor for TypedValue.


def _from_numpy(a):
    assert a.dtype.name in ['int64', 'uint64', 'int32',
                            'uint32', 'int16', 'uint16', 'int8', 'uint8', 'bool'], 'Unsupported numpy.dtype'
    f = getattr(cc, 'serialize_to_str_' + a.dtype.name)
    return cc.TypedValue.from_str(f(a))


def _tv_new(_cls, *args):
    """Creates new typed value from serialized string or from numpy array."""
    assert len(args) in [1, 2], 'Unexpected number of args'
    if len(args) == 1:
        a = args[0]
        assert isinstance(a, str) or type(
            a).__name__ == 'ndarray', 'Argument must be str or numpy.ndarray'
        if isinstance(a, str):
            return cc.TypedValue.from_str(a)
        else:
            return _from_numpy(a)
    return cc.TypedValue.from_type_and_value(args[0], args[1])


TypedValue = cc.TypedValue
TypedValue.from_numpy = _from_numpy
TypedValue.__new__ = _tv_new


def get_slice(self, array_slice):
    if isinstance(array_slice, slice):
        array_slice = (array_slice,)
    assert isinstance(array_slice, tuple)
    assert all(isinstance(element, slice) or isinstance(element, int) or (element == Ellipsis)
               for element in array_slice)
    internal_slice = []
    for element in array_slice:
        if isinstance(element, int):
            internal_slice.append(cc.SliceElement.from_single_element(element))
        elif element == Ellipsis:
            internal_slice.append(cc.SliceElement.from_ellipsis())
        else:
            internal_slice.append(
                cc.SliceElement.from_sub_array(element.start, element.stop, element.step))
    return self.get_slice(internal_slice)


Node.__getitem__ = get_slice

# Context managers.


def _noop(self):
    pass


def _finalize(self, type, value, traceback):
    if type != None:
        return False
    self.finalize()


Graph.__exit__ = _finalize
Graph.__enter__ = _noop
Context.__exit__ = _finalize
Context.__enter__ = _noop

Node.__doc__ = cc.Node.__doc__
Graph.__doc__ = cc.Graph.__doc__
Context.__doc__ = cc.Context.__doc__
