This is the Python wrapper for CipherCore base library.
It currently supports building CipherCore computation graphs.
See https://github.com/ciphermodelabs/ciphercore/ for more details.

Example usage:

```python
import ciphercore as cc

c = cc.create_context()
with c:
  g = c.create_graph()
  with g:
    a = g.input(cc.array_type([10, 20, 30], cc.INT32))
    s = a[..., 2:3] 
    b = s.sum([0, 2])
    b.set_as_output()
  print(g)
  g.set_as_main()
print(c)

```
