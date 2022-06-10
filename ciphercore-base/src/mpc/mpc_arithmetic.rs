use crate::custom_ops::CustomOperationBody;
use crate::data_types::Type;
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, NodeAnnotation, Operation};
use crate::mpc::mpc_compiler::{check_private_tuple, get_zero_shares, PARTIES};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct AddMPC {}

#[typetag::serde]
impl CustomOperationBody for AddMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // Panics since:
        // - the user has no direct access to this function.
        // - the MPC compiler should pass the correct number of arguments
        // and this panic should never happen.
        if argument_types.len() != 2 {
            panic!("AddMPC should have two inputs");
        }
        let g = context.create_graph()?;
        let t0 = argument_types[0].clone();
        let t1 = argument_types[1].clone();
        let i0 = g.input(t0.clone())?;
        let i1 = g.input(t1.clone())?;

        // If an input is private, i.e. a tuple of 3 elements (a0, a1, a2), then
        // the parties can access the following elements:
        // 1st party -> a0, a1;
        // 2nd party -> a1, a2;
        // 3rd party -> a2, a0.
        let adder = |l_node: Node, r_node: Node, is_r_private: bool| -> Result<Node> {
            let mut outputs = vec![];
            for i in 0..PARTIES as u64 {
                let a0i = g.tuple_get(l_node.clone(), i)?;
                let a = if is_r_private {
                    let a1i = g.tuple_get(r_node.clone(), i)?;
                    g.add(a0i, a1i)?
                } else if i == 0 {
                    g.add(a0i, r_node.clone())?
                } else {
                    let r_type = r_node.get_type()?;
                    let zero = g.constant(r_type.clone(), Value::zero_of_type(r_type))?;
                    g.add(a0i, zero)?
                };
                outputs.push(a);
            }
            g.create_tuple(outputs)?.set_as_output()
        };

        match (t0, t1) {
            (Type::Tuple(v0), Type::Tuple(v1)) => {
                check_private_tuple(v0)?;
                check_private_tuple(v1)?;
                adder(i0, i1, true)?;
            }
            (Type::Tuple(v0), Type::Array(_, _) | Type::Scalar(_)) => {
                check_private_tuple(v0)?;
                adder(i0, i1, false)?;
            }
            (Type::Array(_, _) | Type::Scalar(_), Type::Tuple(v1)) => {
                check_private_tuple(v1)?;
                adder(i1, i0, false)?;
            }
            (Type::Array(_, _) | Type::Scalar(_), Type::Array(_, _) | Type::Scalar(_)) => {
                g.add(i0, i1)?.set_as_output()?;
            }
            _ => {
                panic!("Inconsistency with type checker");
            }
        }

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "AddMPC".to_owned()
    }
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct SubtractMPC {}

#[typetag::serde]
impl CustomOperationBody for SubtractMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // Panics since:
        // - the user has no direct access to this function.
        // - the MPC compiler should pass the correct number of arguments
        // and this panic should never happen.
        if argument_types.len() != 2 {
            panic!("SubtractMPC should have two inputs");
        }
        let g = context.create_graph()?;
        let t0 = argument_types[0].clone();
        let t1 = argument_types[1].clone();
        let i0 = g.input(t0.clone())?;
        let i1 = g.input(t1.clone())?;

        // If an input is private, i.e. a tuple of 3 elements (a0, a1, a2), then
        // the parties can access the following elements:
        // 1st party -> a0, a1;
        // 2nd party -> a1, a2;
        // 3rd party -> a2, a0.

        match (t0.clone(), t1.clone()) {
            (Type::Tuple(v0), Type::Tuple(v1)) => {
                check_private_tuple(v0)?;
                check_private_tuple(v1)?;
                let mut outputs = vec![];
                for i in 0..PARTIES as u64 {
                    let a0i = g.tuple_get(i0.clone(), i)?;
                    let a1i = g.tuple_get(i1.clone(), i)?;
                    outputs.push(a0i.subtract(a1i)?);
                }
                g.create_tuple(outputs)?.set_as_output()?;
            }
            (Type::Tuple(v0), Type::Array(_, _) | Type::Scalar(_)) => {
                check_private_tuple(v0)?;
                let mut outputs = vec![];
                let zero = g.constant(t1.clone(), Value::zero_of_type(t1))?;
                for i in 0..PARTIES as u64 {
                    let a0i = g.tuple_get(i0.clone(), i)?;
                    let share = if i == 0 {
                        a0i.subtract(i1.clone())?
                    } else {
                        a0i.subtract(zero.clone())?
                    };
                    outputs.push(share);
                }
                g.create_tuple(outputs)?.set_as_output()?;
            }
            (Type::Array(_, _) | Type::Scalar(_), Type::Tuple(v1)) => {
                check_private_tuple(v1)?;
                let mut outputs = vec![];
                let zero = g.constant(t0.clone(), Value::zero_of_type(t0))?;
                for i in 0..PARTIES as u64 {
                    let a1i = g.tuple_get(i1.clone(), i)?;
                    let share = if i == 0 {
                        i0.subtract(a1i)?
                    } else {
                        zero.subtract(a1i)?
                    };
                    outputs.push(share);
                }
                g.create_tuple(outputs)?.set_as_output()?;
            }
            (Type::Array(_, _) | Type::Scalar(_), Type::Array(_, _) | Type::Scalar(_)) => {
                g.subtract(i0, i1)?.set_as_output()?;
            }
            _ => {
                panic!("Inconsistency with type checker");
            }
        }

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "SubtractMPC".to_owned()
    }
}

fn bilinear_product(l: Node, r: Node, op: Operation) -> Result<Node> {
    match op {
        Operation::Multiply => l.multiply(r),
        Operation::Dot => l.dot(r),
        Operation::Matmul => l.matmul(r),
        _ => Err(runtime_error!("Not a bilinear product")),
    }
}
/// Given two nodes containing public and private values, applies a bilinear product to them.
/// swap_flag indicates whether the first node is public (true) or private (false).
fn mixed_product(
    node0: Node,
    node1: Node,
    g: Graph,
    op: Operation,
    swap_flag: bool,
) -> Result<Node> {
    let mut outputs = vec![];
    for i in 0..PARTIES as u64 {
        let (l, r) = if swap_flag {
            let share = g.tuple_get(node1.clone(), i)?;
            (node0.clone(), share)
        } else {
            let share = g.tuple_get(node0.clone(), i)?;
            (share, node1.clone())
        };
        let res = bilinear_product(l, r, op.clone())?;
        outputs.push(res);
    }
    g.create_tuple(outputs)?.set_as_output()
}

/// Given two nodes containing private values, applies a bilinear product to them
/// using an interactive MPC protocol from ABY3.
fn private_product(
    node0: Node,
    node1: Node,
    prf_type: Type,
    g: Graph,
    op: Operation,
) -> Result<Node> {
    let mut outputs = vec![];
    let prf_keys = g.input(prf_type)?;
    let mut shares0 = vec![];
    let mut shares1 = vec![];
    for i in 0..PARTIES as u64 {
        let share0 = g.tuple_get(node0.clone(), i)?;
        shares0.push(share0);
        let share1 = g.tuple_get(node1.clone(), i)?;
        shares1.push(share1);
    }
    let mut z_shares = vec![];
    for i in 0..PARTIES {
        let ip1 = (i + 1) % PARTIES;
        // secret shares from node0 = (x_0, x_1, x_2)
        // secret shares from node1 = (y_0, y_1, y_2)
        // y_i + y_(i+1)
        let z1 = g.add(shares1[i].clone(), shares1[ip1].clone())?;
        // x_i * y_i + x_i * y_(i+1)
        let z2 = bilinear_product(shares0[i].clone(), z1, op.clone())?;
        // z3 = x_(i+1) * y_i
        let z3 = bilinear_product(shares0[ip1].clone(), shares1[i].clone(), op.clone())?;

        // z = x_i * y_i + x_i * y_(i+1) + x_(i+1) * y_i
        let z = g.add(z2, z3)?;
        z_shares.push(z.clone());
    }
    let zero_shares = get_zero_shares(g.clone(), prf_keys, z_shares[0].get_type()?)?;

    for i in 0..PARTIES {
        // x_i * y_i + x_i * y_(i+1) + x_(i+1) * y_i + zero_share_i
        let mul_share = g.add(z_shares[i].clone(), zero_shares[i].clone())?;
        // networking
        let network_node = g.nop(mul_share)?;
        let im1 = ((i + PARTIES - 1) % PARTIES) as u64;
        network_node.add_annotation(NodeAnnotation::Send(i as u64, im1))?;
        outputs.push(network_node);
    }
    g.create_tuple(outputs)?.set_as_output()
}

fn instantiate_bilinear_product(
    context: Context,
    argument_types: Vec<Type>,
    op: Operation,
) -> Result<Graph> {
    let op_name = match op {
        Operation::Multiply => "MultiplyMPC".to_owned(),
        Operation::Dot => "DotMPC".to_owned(),
        Operation::Matmul => "MatmulMPC".to_owned(),
        _ => return Err(runtime_error!("Not a bilinear product")),
    };
    // Panics since:
    // - the user has no direct access to this function.
    // - the MPC compiler should pass the correct number of arguments
    // and this panic should never happen.
    if !(2..=3).contains(&argument_types.len()) {
        panic!("{} should have either 2 or 3 inputs.", op_name);
    }
    let g = context.create_graph()?;
    let t0 = argument_types[0].clone();
    let t1 = argument_types[1].clone();
    let i0 = g.input(t0.clone())?;
    let i1 = g.input(t1.clone())?;

    // If an input is private, i.e. a tuple of 3 elements (a0, a1, a2), then
    // the parties can access the following elements:
    // 1st party -> a0, a1;
    // 2nd party -> a1, a2;
    // 3rd party -> a2, a0.
    match (t0, t1) {
        (Type::Tuple(v0), Type::Tuple(v1)) => {
            check_private_tuple(v0)?;
            check_private_tuple(v1)?;
            // Panics since:
            // - the user has no direct access to this function,
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            if argument_types.len() != 3 {
                panic!(
                    "{} with two private inputs should be provided a tuple of keys",
                    op_name
                );
            }
            let prf_type = argument_types[2].clone();
            private_product(i0, i1, prf_type, g.clone(), op)?;
        }
        (Type::Tuple(v0), Type::Array(_, _) | Type::Scalar(_)) => {
            check_private_tuple(v0)?;
            mixed_product(i0, i1, g.clone(), op, false)?;
        }
        (Type::Array(_, _) | Type::Scalar(_), Type::Tuple(v1)) => {
            check_private_tuple(v1)?;
            mixed_product(i0, i1, g.clone(), op, true)?;
        }
        (Type::Array(_, _) | Type::Scalar(_), Type::Array(_, _) | Type::Scalar(_)) => {
            let o = bilinear_product(i0, i1, op)?;
            o.set_as_output()?;
        }
        _ => {
            panic!("Inconsistency with type checker");
        }
    }
    g.finalize()?;
    Ok(g)
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct MultiplyMPC {}

#[typetag::serde]
impl CustomOperationBody for MultiplyMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        instantiate_bilinear_product(context, argument_types, Operation::Multiply)
    }

    fn get_name(&self) -> String {
        "MultiplyMPC".to_owned()
    }
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct DotMPC {}

#[typetag::serde]
impl CustomOperationBody for DotMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        instantiate_bilinear_product(context, argument_types, Operation::Dot)
    }

    fn get_name(&self) -> String {
        "DotMPC".to_owned()
    }
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct MatmulMPC {}

#[typetag::serde]
impl CustomOperationBody for MatmulMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        instantiate_bilinear_product(context, argument_types, Operation::Matmul)
    }

    fn get_name(&self) -> String {
        "MatmulMPC".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::subtract_vectors_u64;
    use crate::data_types::{array_type, tuple_type, ArrayShape, ScalarType, INT32, UINT32};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::inline::inline_ops::{InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{prepare_for_mpc_evaluation, IOStatus};

    fn prepare_arithmetic_context(
        input_party_map: Vec<IOStatus>,
        output_parties: Vec<IOStatus>,
        op: Operation,
        st: ScalarType,
        dims: Vec<ArrayShape>,
    ) -> Result<Context> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let mut types = vec![];
        for shape in dims {
            types.push(array_type(shape, st.clone()));
        }
        let i1 = g.input(types[0].clone())?;
        i1.set_name("Input 1")?;
        let i2 = g.input(types[1].clone())?;
        i2.set_name("Input 2")?;
        let o = match op {
            Operation::Add => {
                let a1 = i1.add(i2)?;
                a1.add(g.input(types[2].clone())?)?
            }
            Operation::Subtract => {
                let a1 = i1.subtract(i2)?;
                a1.subtract(g.input(types[2].clone())?)?
            }
            Operation::Multiply | Operation::Dot | Operation::Matmul => {
                let a1 = bilinear_product(i1, i2, op.clone())?;
                bilinear_product(a1, g.input(types[2].clone())?, op)?
            }
            _ => {
                panic!("Shouldn't be here");
            }
        };
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g)?;
        c.finalize()?;

        let inline_config = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };

        let mpc_c = prepare_for_mpc_evaluation(
            c,
            vec![input_party_map],
            vec![output_parties],
            inline_config,
        )?;
        // Check names
        let mpc_graph = mpc_c.get_main_graph()?;
        assert!(mpc_c.retrieve_node(mpc_graph.clone(), "Input 1").is_ok());
        assert!(mpc_c.retrieve_node(mpc_graph, "Input 2").is_ok());
        Ok(mpc_c)
    }

    fn prepare_arithmetic_input(
        input: Vec<Vec<u64>>,
        input_status: Vec<IOStatus>,
        st: ScalarType,
    ) -> Result<Vec<Value>> {
        let mut res = vec![];
        for i in 0..3 {
            let n = input[i].len();
            if input_status[i] == IOStatus::Shared {
                let mut v = vec![];
                // shares of input = (input-3, 1, 2)
                if n == 0 {
                    panic!("Use non-empty input");
                }
                let threes = vec![3; n];
                let first_share = subtract_vectors_u64(&input[i], &threes, st.get_modulus())?;
                v.push(Value::from_flattened_array(&first_share, st.clone())?);
                for j in 1..PARTIES {
                    let share = vec![j; n];
                    v.push(Value::from_flattened_array(&share, st.clone())?);
                }
                res.push(Value::from_vector(v));
            } else {
                res.push(Value::from_flattened_array(&input[i], st.clone())?);
            }
        }
        Ok(res)
    }

    fn check_arithmetic_output(
        mpc_graph: Graph,
        inputs: Vec<Value>,
        expected: Vec<u64>,
        st: ScalarType,
        dims: ArrayShape,
        output_parties: Vec<IOStatus>,
    ) -> Result<()> {
        let output = random_evaluate(mpc_graph.clone(), inputs)?;
        let output_type = array_type(dims.clone(), st.clone());

        if output_parties.is_empty() {
            // check that mpc_output is a sharing of plain_output
            assert!(output.check_type(tuple_type(vec![output_type.clone(); PARTIES]))?);
            // check that output is a sharing of expected
            let out = output.access_vector(|v| {
                let flat_dims: u64 = dims.iter().product();
                let mut res = vec![0; flat_dims as usize];
                for val in v {
                    let arr = val.to_flattened_array_u64(output_type.clone())?;
                    for i in 0..flat_dims {
                        res[i as usize] += arr[i as usize];
                    }
                }
                if let Some(m) = st.get_modulus() {
                    Ok(res.iter().map(|x| x % m).collect())
                } else {
                    Ok(res)
                }
            })?;
            assert_eq!(out, expected)
        } else {
            assert!(output.check_type(output_type.clone())?);
            assert_eq!(output.to_flattened_array_u64(output_type)?, expected);
        }
        Ok(())
    }

    fn helper_add_subtract(
        op: Operation,
        st: ScalarType,
        input: Vec<Vec<u64>>,
        expected: Vec<u64>,
        dims_in: Vec<ArrayShape>,
        dims_out: ArrayShape,
    ) -> Result<()> {
        let helper = |op: Operation,
                      st: ScalarType,
                      input: Vec<Vec<u64>>,
                      expected: Vec<u64>,
                      input_status: Vec<IOStatus>,
                      output_parties: Vec<IOStatus>|
         -> Result<()> {
            let mpc_context = prepare_arithmetic_context(
                input_status.clone(),
                output_parties.clone(),
                op,
                st.clone(),
                dims_in.clone(),
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            let inputs = prepare_arithmetic_input(input, input_status, st.clone())?;

            check_arithmetic_output(
                mpc_graph,
                inputs,
                expected,
                st,
                dims_out.clone(),
                output_parties,
            )?;

            Ok(())
        };

        helper(
            op.clone(),
            st.clone(),
            input.clone(),
            expected.clone(),
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            vec![IOStatus::Party(0)],
        )?;
        helper(
            op.clone(),
            st.clone(),
            input.clone(),
            expected.clone(),
            vec![IOStatus::Public, IOStatus::Party(0), IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )?;
        helper(
            op.clone(),
            st.clone(),
            input.clone(),
            expected.clone(),
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Party(0)],
            vec![IOStatus::Party(0), IOStatus::Party(1)],
        )?;
        helper(
            op.clone(),
            st.clone(),
            input.clone(),
            expected.clone(),
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )?;
        helper(
            op.clone(),
            st.clone(),
            input.clone(),
            expected.clone(),
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            vec![],
        )?;
        helper(
            op,
            st,
            input,
            expected,
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![],
        )?;
        Ok(())
    }

    #[test]
    fn test_add() {
        helper_add_subtract(
            Operation::Add,
            UINT32,
            vec![vec![2, 3], vec![4, 5], vec![6, 7]],
            vec![12, 15],
            vec![vec![2]; 3],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Add,
            UINT32,
            vec![vec![2, 3], vec![4], vec![6]],
            vec![12, 13],
            vec![vec![2], vec![1], vec![1]],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Add,
            UINT32,
            vec![vec![2], vec![3, 4], vec![6]],
            vec![11, 12],
            vec![vec![1], vec![2], vec![1]],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Add,
            UINT32,
            vec![vec![2], vec![3], vec![4, 6]],
            vec![9, 11],
            vec![vec![1], vec![1], vec![2]],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Add,
            INT32,
            vec![vec![1, 2], vec![(1 << 32) - 4, (1 << 32) - 5], vec![1, 2]],
            vec![(1 << 32) - 2, (1 << 32) - 1],
            vec![vec![2]; 3],
            vec![2],
        )
        .unwrap();
    }

    #[test]
    fn test_subtract() {
        helper_add_subtract(
            Operation::Subtract,
            UINT32,
            vec![vec![10, 9], vec![4, 5], vec![5, 2]],
            vec![1, 2],
            vec![vec![2]; 3],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Subtract,
            UINT32,
            vec![vec![10, 9], vec![4], vec![5]],
            vec![1, 0],
            vec![vec![2], vec![1], vec![1]],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Subtract,
            UINT32,
            vec![vec![10], vec![3, 4], vec![5]],
            vec![2, 1],
            vec![vec![1], vec![2], vec![1]],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Subtract,
            UINT32,
            vec![vec![10], vec![4], vec![2, 5]],
            vec![4, 1],
            vec![vec![1], vec![1], vec![2]],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Subtract,
            INT32,
            vec![vec![10, 9], vec![4, 5], vec![7, 6]],
            vec![(1 << 32) - 1, (1 << 32) - 2],
            vec![vec![2]; 3],
            vec![2],
        )
        .unwrap();
    }

    fn bilinear_product_helper(op: Operation, dims: ArrayShape) -> Result<()> {
        let helper = |input_status: Vec<IOStatus>, output_parties: Vec<IOStatus>| -> Result<()> {
            let st = INT32;
            let mpc_context = prepare_arithmetic_context(
                input_status.clone(),
                output_parties.clone(),
                op.clone(),
                st.clone(),
                vec![dims.clone(); 3],
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            let flat_dims: u64 = dims.iter().product();
            let inputs = prepare_arithmetic_input(
                vec![
                    (2..2 + flat_dims).collect(),
                    (4..4 + flat_dims).collect(),
                    (6..6 + flat_dims).collect(),
                ],
                input_status,
                st.clone(),
            )?;

            let expected = match op.clone() {
                Operation::Multiply => vec![48, 105],
                Operation::Dot => vec![138, 161],
                Operation::Matmul => vec![404, 461, 716, 817],
                _ => panic!("Not a bilinear operation"),
            };

            check_arithmetic_output(
                mpc_graph,
                inputs,
                expected,
                st,
                dims.clone(),
                output_parties,
            )?;

            Ok(())
        };

        helper(
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            vec![IOStatus::Party(0)],
        )
        .unwrap();
        helper(
            vec![IOStatus::Public, IOStatus::Party(0), IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )
        .unwrap();
        helper(
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Party(0)],
            vec![IOStatus::Party(0), IOStatus::Party(1)],
        )
        .unwrap();
        helper(
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )
        .unwrap();
        helper(
            vec![IOStatus::Public, IOStatus::Party(0), IOStatus::Public],
            vec![],
        )
        .unwrap();
        helper(
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![],
        )
        .unwrap();

        Ok(())
    }
    #[test]
    fn test_multiply() {
        bilinear_product_helper(Operation::Multiply, vec![2]).unwrap();
    }

    #[test]
    fn test_dot() {
        bilinear_product_helper(Operation::Dot, vec![2]).unwrap();
    }

    #[test]
    fn test_matmul() {
        bilinear_product_helper(Operation::Matmul, vec![2, 2]).unwrap();
    }
}
