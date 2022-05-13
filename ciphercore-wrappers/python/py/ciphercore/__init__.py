import ciphercore_native as cc
import types

# Re-export scalar types.
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
Type = cc.Type
ScalarType = cc.ScalarType
array_type = cc.array_type
scalar_type = cc.scalar_type
tuple_type = cc.tuple_type
vector_type = cc.vector_type
named_tuple_type = cc.named_tuple_type

# Re-export graph-related primitives.
Context = cc.Context
Graph = cc.Graph
Node = cc.Node
create_context = cc.create_context

# Syntactic sugar: overriding operators.
Node.__add__ = Node.add
Node.__sub__ = Node.subtract
Node.__mul__ = Node.multiply
Node.__matmul__ = Node.matmul
Node.__and__ = Node.multiply
Node.__xor__ = Node.multiply

# Slicing is a bit trickier.
def _get_slice(self, array_slice):
  if isinstance(array_slice, slice):
    array_slice = (array_slice,)
  assert isinstance(array_slice, tuple)
  assert all(isinstance(element, slice) or isinstance(element, int) or (element == Ellipsis)
             for element in array_slice)
  internal_slice = []
  for element in array_slice:
    if isinstance(element, int):
      internal_slice.append(
          cc.SliceElement(cc.SliceElement.Kind.SingleIndex,
                          cc.MaybeInt64(True, element),
                          cc.MaybeInt64(False, 0), cc.MaybeInt64(False, 0)))
    elif element == Ellipsis:
      internal_slice.append(
          cc.SliceElement(cc.SliceElement.Kind.Ellipsis,
                          cc.MaybeInt64(False, 0), cc.MaybeInt64(False, 0), cc.MaybeInt64(False, 0)))
    else:
      internal_slice.append(
              cc.SliceElement(cc.SliceElement.Kind.SubArray,
                              cc.MaybeInt64(True, element.start) if element.start is not None else cc.MaybeInt64(False, 0),
                              cc.MaybeInt64(True, element.stop) if element.stop is not None else cc.MaybeInt64(False, 0),
                              cc.MaybeInt64(True, element.step) if element.step is not None else cc.MaybeInt64(False, 0)))
  return self.get_slice(internal_slice)

Node.__getitem__ = _get_slice


# Cosmetics.
def _node_repr(self):
  return 'Node[type={}]'.format(self.get_type().__repr__())

Node.__repr__ = _node_repr

def _graph_repr(self):
  return 'Graph[num_nodes={}]'.format(len(self.get_nodes()))

Graph.__repr__ = _graph_repr


# Context managers.
def _noop(self):
  pass

def _finalize(self, *args):
  self.finalize()

Graph.__exit__ = _finalize
Graph.__enter__ = _noop
Context.__exit__ = _finalize
Context.__enter__ = _noop

