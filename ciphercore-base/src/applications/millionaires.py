import ciphercore as cc

# Create a context
c = cc.create_context()
with c:
    # Create a graph in a given context that will be used for matrix multiplication
    g = c.create_graph()
    with g:
        # For each millionaire, add input nodes to the empty graph g created above.
        # Input nodes are instantiated with the single unsigned 32-bit integer.
        # This should be enough to represent the wealth of each millionaire.
        first_millionaire = g.input(cc.scalar_type(cc.UINT32))
        second_millionaire = g.input(cc.scalar_type(cc.UINT32))

        # For each millionaire, convert integer value to binary array in order to perform comparison.
        # Add custom operation to the graph specifying the custom operation and its arguments: `first_millionaire` and `second_millionaire`.
        # This operation will compute the bit `(first_millionaire > second_millionaire)`.
        output = first_millionaire.a2b() > second_millionaire.a2b()

        # Before computation, every graph should be finalized, which means that it should have a designated output node.
        # This can be done by calling `g.set_output_node(output)?` or as below.
        output.set_as_output()
    # Set this graph as main to be able to finalize the context
    g.set_as_main()
# Serialize the context and print it to stdout
print(c)
