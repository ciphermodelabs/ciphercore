import ciphercore as cc

# Number of rows of the first matrix
n = 2
# Number of columns of the first matrix (and number of rows of the second matrix)
m = 3
# Number of columns of the second matrix
k = 4

# Scalar type of matrix elements
st = cc.INT32

# Create a context
c = cc.create_context()
with c:
    # Create a graph in a given context that will be used for matrix multiplication
    g = c.create_graph()
    with g:
        # Create types of input matrices.
        # Matrices can be represented as arrays with two 2-dimensional shapes.
        # First, create the array type of a first matrix with shape `[n, m]`, which corresponds to a (n x m)-matrix.
        first_matrix_type = cc.array_type([n, m], st)
        # Second, create the array type of a second matrix with shape `[m, k]`, which corresponds to a (m x k)-matrix.
        second_matrix_type = cc.array_type([m, k], st)
        
        # For each input matrix, add input nodes to the empty graph g created above.
        # Input nodes require the types of input matrices generated in previous lines.
        first_matrix_input = g.input(first_matrix_type)
        second_matrix_input = g.input(second_matrix_type)

        # Matrix multiplication is a built-in function of CipherCore, so it can be computed by a single computational node.
        output = first_matrix_input @ second_matrix_input

        # Before computation, every graph should be finalized, which means that it should have a designated output node.
        # This can be done by calling `g.set_output_node(output)?` or as below.
        output.set_as_output()
    # Set this graph as main to be able to finalize the context
    g.set_as_main()
# Serialize the context and print it to stdout
print(c)
