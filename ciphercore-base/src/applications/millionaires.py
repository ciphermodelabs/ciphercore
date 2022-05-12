import ciphercore as cc

# Create a context
c = cc.create_context()
with c:
    # Create a graph in a given context that will be used for matrix multiplication
    g = c.create_graph()
    with g:
        # For each millionaire, add input nodes to the empty graph g created above.
        # Input nodes are instantiated with binary arrays of 32 bits.
        # This should be enough to represent the wealth of each millionaire.
        first_millionaire = g.input(cc.array_type([32], cc.BIT))
        second_millionaire = g.input(cc.array_type([32], cc.BIT))

        # Millionaires' problem boils down to computing the greater-than (>) function.
        # In CipherCore, comparison functions are realized via custom operations,
        # which are a special kind of operations that accept varying number of inputs and input types.
        # To add a custom operation node to the graph, create it first.
        # Note that the GreaterThan custom operation has a Boolean parameter that indicates whether input binary arrays represent signed integers
        op = '{"body":{"type":"GreaterThan","signed_comparison":false}}'
        # Add custom operation to the graph specifying the custom operation and its arguments: `first_millionaire` and `second_millionaire`.
        # This operation will compute the bit `(first_millionaire > second_millionaire)`.
        output = g.custom_op(op, [first_millionaire, second_millionaire])

        # Before computation, every graph should be finalized, which means that it should have a designated output node.
        # This can be done by calling `g.set_output_node(output)?` or as below.
        output.set_as_output()
    # Set this graph as main to be able to finalize the context
    g.set_as_main()
# Serialize the context and print it to stdout
print(c)
