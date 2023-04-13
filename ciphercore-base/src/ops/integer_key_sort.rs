//! Sort by integer key.
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{ScalarType, Type, BIT};
use crate::errors::Result;
use crate::graphs::*;

use serde::{Deserialize, Serialize};

use super::comparisons::flip_msb;
use super::utils::unsqueeze;

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct SortByIntegerKey {
    pub key: String,
}

#[typetag::serde]
impl CustomOperationBody for SortByIntegerKey {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!("SortByIntegerKey accepts only 1 argument"));
        }
        if !arguments_types[0].is_named_tuple() {
            return Err(runtime_error!(
                "SortByIntegerKey accepts only named tuple as input"
            ));
        }
        let g = context.create_graph()?;
        let input = g.input(arguments_types[0].clone())?;
        let mut elements = vec![];
        for (name, _) in arguments_types[0].get_named_types()? {
            let node = input.named_tuple_get(name.clone())?;
            if name == self.key {
                elements.push((name, integer_to_bits(node)?));
            } else {
                elements.push((name, node));
            }
        }
        let sorted = g.create_named_tuple(elements)?.sort(self.key.clone())?;
        let mut elements = vec![];
        for (name, t) in arguments_types[0].get_named_types()? {
            let node = sorted.named_tuple_get(name.clone())?;
            if name == self.key {
                elements.push((name, integer_from_bits(node, t.get_scalar_type())?));
            } else {
                elements.push((name, node));
            }
        }
        let output = g.create_named_tuple(elements)?;
        g.set_output_node(output)?;
        g.finalize()?;
        Ok(g)
    }
    fn get_name(&self) -> String {
        "SortIntegers".to_string()
    }
}

fn integer_to_bits(node: Node) -> Result<Node> {
    let st = node.get_type()?.get_scalar_type();
    if st == BIT {
        return unsqueeze(node, -1);
    }
    // Convert to bit representation.
    let node = node.a2b()?;
    // We need to preocess sign bit correctly. Reuse method from `comparisons`.
    let node = if st.is_signed() {
        flip_msb(node)?
    } else {
        node
    };
    // Reverse order of bits.
    node.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(None, None, Some(-1)),
    ])
}

fn integer_from_bits(node: Node, st: ScalarType) -> Result<Node> {
    if st == BIT {
        return node.get_slice(vec![SliceElement::Ellipsis, SliceElement::SingleIndex(0)]);
    }
    // Order of bits was reversed.
    let node = node.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(None, None, Some(-1)),
    ])?;
    // We need to preocess sign bit correctly. Reuse method from `comparisons`.
    let node = if st.is_signed() {
        flip_msb(node)?
    } else {
        node
    };
    // Convert to integer representation.
    node.b2a(st)
}

#[cfg(test)]
mod tests {
    use crate::{
        custom_ops::{run_instantiation_pass, CustomOperation},
        data_types::{INT64, UINT64},
        evaluators::random_evaluate,
        graphs::util::simple_context,
        typed_value::TypedValue,
        typed_value_operations::{TypedValueArrayOperations, TypedValueOperations},
    };

    use super::*;

    fn test_helper(mut x: Vec<i64>, st: ScalarType) -> Result<()> {
        let values = TypedValue::from_ndarray(ndarray::Array::from(x.clone()).into_dyn(), st)?;
        let c = simple_context(|g| {
            let input = g.input(values.get_type())?;
            let key = "key".to_string();
            let node = g.create_named_tuple(vec![(key.clone(), input)])?;
            let sorted = g.custom_op(
                CustomOperation::new(SortByIntegerKey { key: key.clone() }),
                vec![node],
            )?;
            sorted.named_tuple_get(key)
        })?;
        let output_type = c.get_main_graph()?.get_output_node()?.get_type()?;
        let c = run_instantiation_pass(c)?.context;

        let result = random_evaluate(c.get_main_graph()?, vec![values.value])?
            .to_flattened_array_i64(output_type)?;
        x.sort();
        assert_eq!(result, x);
        Ok(())
    }

    #[test]
    fn test_correctnes() -> Result<()> {
        let x = vec![50, 1, 300, 13, 74532, 100];
        test_helper(x, UINT64)?;
        let x = vec![12, 45, -1, 0, 2, -100, 50, 1, 300];
        test_helper(x, INT64)?;
        let x = vec![0, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        test_helper(x, BIT)
    }
}
