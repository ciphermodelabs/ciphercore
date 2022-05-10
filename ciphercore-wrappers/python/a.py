import ciphercore as cc

c = cc.create_context()
print(c.get_graphs())
with c:
  g = c.create_graph()
  with g:
    a = g.input(cc.array_type([10, 20, 30], cc.INT32))
    b = a.sum([0, 2])
    s = a[..., 2:3]
    n = g.create_named_tuple([('boda',b)])
    s.set_as_output()
    print(a.get_operation())
    print(s)
    print(c.get_graphs())
    print(g.get_nodes())
    print(b.get_operation())
    print(b.get_global_id())
    print(n.get_operation())
    print(s.get_operation())
  print(g)
  g.set_as_main()
print(c)
