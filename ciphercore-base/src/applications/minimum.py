import ciphercore as cc

# Number of elements of an array (i.e., 2^n)
n = 4
# Scalar type of array elements; it should be unsigned, i.e. BIT, UINT8, UINT16, UINT32 or UINT64
st = cc.UINT32

# Create a context
c = cc.create_context()
with c:
    # Create a graph in a given context that will be used for matrix multiplication
    g = c.create_graph()
    with g:
        # Create the type of the input array with `n` elements.
        # To find the minimum of an array, we resort to the custom operation Min (see ops.rs) that accepts only binary input.
        b = st.size_in_bits()
        input_type = cc.array_type([2 ** n, b], cc.BIT)

        # Add an input node to the empty graph g created above.
        # This input node requires the input array type generated previously.
        binary_array = g.input(input_type)

        # We find the minimum using the tournament method. This allows to reduce the graph size to O(n) from O(2^n) nodes.
        # Namely, we split the input array into pairs, find a minimum within each pair and create a new array from these minima.
        # Then, we repeat this procedure for the new array.
        # For example, let [2,7,0,3,11,5,0,4] be an input array.
        # The 1st iteration yields [min(2,11), min(7,5), min(0,0), min(3,4)] = [2,5,0,3]
        # The 2nd iteration results in [min(2,0), min(5,3)] = [0,3]
        # The 3rd iteration returns [min(0,3)] = [0]
        for level in reversed(range(n)):
            # Extract the first half of the array using the [Graph::get_slice] operation.
            # Our slicing conventions follow [the NumPy rules](https://numpy.org/doc/stable/user/basics.indexing.html).
            half1 = binary_array[:(2 ** level)]
            # Extract the first half of the array using the [Graph::get_slice] operation.
            half2 = binary_array[(2 ** level):]
            # Compare the first half with the second half elementwise to find minimums.
            # This is done via the custom operation Min (see ops.rs).
            binary_array = g.custom_op('{"body":{"type":"Min"}}', [half1, half2])
        # Convert output from the binary form to the arithmetic form
        output = binary_array
        if st != cc.BIT:
            output = binary_array.b2a(st)
        output.set_as_output()
    # Set this graph as main to be able to finalize the context
    g.set_as_main()
# Serialize the context and print it to stdout
print(c)
