//! [Millionaires' problem](https://en.wikipedia.org/wiki/Yao%27s_Millionaires%27_problem).
//! Two millionaires want to find out who is richer without revealing their wealth.  
use crate::custom_ops::CustomOperation;
use crate::data_types::{scalar_type, UINT32};
use crate::errors::Result;
use crate::graphs::{Context, Graph};
use crate::ops::comparisons::GreaterThan;

/// Creates a graph that solves Millionaires' problem.
///
/// It computes a bit that is equal to 1 if the first millionaire is richer than the second one and 0 otherwise.
///
/// # Arguments
///
/// * `context` - context where a Millionaires' problem graph should be created
///
/// # Returns
///
/// Graph that solves Millionaires' problem
pub fn create_millionaires_graph(context: Context) -> Result<Graph> {
    // Create a graph in a given context that will be used for Millionaires' problem
    let g = context.create_graph()?;

    // For each millionaire, add input nodes to the empty graph g created above.
    // Input nodes are instantiated with binary arrays of 32 bits.
    // This should be enough to represent the wealth of each millionaire.
    let first_millionaire = g.input(scalar_type(UINT32))?;
    let second_millionaire = g.input(scalar_type(UINT32))?;

    // Millionaires' problem boils down to computing the greater-than (>) function.
    // In CipherCore, comparison functions are realized via custom operations,
    // which are a special kind of operations that accept varying number of inputs and input types.
    // To add a custom operation node to the graph, create it first.
    // Note that the GreaterThan custom operation has a Boolean parameter that indicates whether input binary arrays represent signed integers
    let op = CustomOperation::new(GreaterThan {
        signed_comparison: false,
    });
    // For each millionaire, convert integer value to binary array in order to perform comparison.
    // Add custom operation to the graph specifying the custom operation and its arguments: `first_millionaire` and `second_millionaire`.
    // This operation will compute the bit `(first_millionaire > second_millionaire)`.
    let output = g.custom_op(
        op,
        vec![first_millionaire.a2b()?, second_millionaire.a2b()?],
    )?;

    // Before computation, every graph should be finalized, which means that it should have a designated output node.
    // This can be done by calling `g.set_output_node(output)?` or as below.
    output.set_as_output()?;
    // Finalization checks that the output node of the graph g is set. After finalization the graph can't be changed.
    g.finalize()?;

    Ok(g)
}

#[cfg(test)]
mod tests {
    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::BIT;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use std::ops::Not;

    use super::*;

    fn test_millionaires_helper<
        T1: TryInto<u128> + Not<Output = T1> + TryInto<u8> + Copy,
        T2: TryInto<u128> + Not<Output = T2> + TryInto<u8> + Copy,
    >(
        input1_value: T1,
        input2_value: T2,
    ) -> Value {
        || -> Result<Value> {
            let c = create_context()?;
            let g = create_millionaires_graph(c.clone())?;
            g.set_as_main()?;
            c.finalize()?;
            let mapped_c = run_instantiation_pass(c)?.get_context();
            let mapped_g = mapped_c.get_main_graph()?;

            let val1 = Value::from_scalar(input1_value, UINT32)?;
            let val2 = Value::from_scalar(input2_value, UINT32)?;
            random_evaluate(mapped_g, vec![val1, val2])
        }()
        .unwrap()
    }

    #[test]
    fn test_matmul() {
        || -> Result<()> {
            assert!(test_millionaires_helper(2000, 1000) == Value::from_scalar(1, BIT)?);
            assert!(test_millionaires_helper(1000, 2000) == Value::from_scalar(0, BIT)?);
            assert!(test_millionaires_helper(1000, 1000) == Value::from_scalar(0, BIT)?);
            Ok(())
        }()
        .unwrap();
    }
}
