use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::Type;
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, NodeAnnotation, Operation};
use crate::mpc::mpc_compiler::{check_private_tuple, PARTIES};
use crate::mpc::utils::ObliviousTransfer;

use serde::{Deserialize, Serialize};

use super::mpc_compiler::is_array_shared;
use super::resharing::reshare;

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct AddMPC {}

#[typetag::serde]
impl CustomOperationBody for AddMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() != 2 {
            return Err(runtime_error!("AddMPC should have two inputs"));
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
                    g.add(a0i, g.zeros(r_node.get_type()?)?)?
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
        if argument_types.len() != 2 {
            return Err(runtime_error!("SubtractMPC should have two inputs"));
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
                let zero = g.zeros(t1)?;
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
                let zero = g.zeros(t0)?;
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
        Operation::MixedMultiply => l.mixed_multiply(r),
        Operation::Gemm(transpose_l, transpose_r) => l.gemm(r, transpose_l, transpose_r),
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
fn private_product(node0: Node, node1: Node, g: Graph, op: Operation) -> Result<Node> {
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
    g.create_tuple(z_shares)?.set_as_output()
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
        Operation::Gemm(_, _) => "GemmMPC".to_owned(),
        _ => return Err(runtime_error!("Not a bilinear product")),
    };
    if argument_types.len() != 2 {
        return Err(runtime_error!(
            "{} should have 2 inputs, {} provided",
            op_name,
            argument_types.len()
        ));
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
            private_product(i0, i1, g.clone(), op)?;
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

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct GemmMPC {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

#[typetag::serde]
impl CustomOperationBody for GemmMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        instantiate_bilinear_product(
            context,
            argument_types,
            Operation::Gemm(self.transpose_a, self.transpose_b),
        )
    }

    fn get_name(&self) -> String {
        format! {"GemmMPC-{}-{}", self.transpose_a, self.transpose_b}
    }
}

// Accepts at least 2 arguments including integer arrays/scalars and binary arrays/scalars that must be multiplied.
// If bits are in the secret shared form, then PRF keys must be provided.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct MixedMultiplyMPC {}

// Mixed multiply of integers known to some party and secret shared bits.
// Let c and b = (b_1, b_2, b_3) be public integers and secret shares of private bits, respectively.
// Let party S be the party knowing a.
// Let party R be the party with index equal to (party S index) - 1.
// Let party H be the party with index equal to (party S index) + 1.
// In this case, the following protocol is applied:
// 1. Parties R and S sample pseudo-random r_s.
// 2. Parties S and H sample pseudo-random r_h.
// 3. Party S  creates two messages using its shares b_s, b_h of b:
//    - m_0 = (b_s XOR b_h) * c - r_s - r_h;
//    - m_1 = NOT(b_s XOR b_h) * c - r_s - r_h.
//    Note that (m_(b_r), r_s, r_h) is the sharing of b * c
// 4. Since parties R and H know b_r and party S knows m_0, m_1,
//    they run the oblivious transfer protocol, which results in party R receiving m_(b_r).
// 5. Party R shares m_(b_r) with party H.
//    As a result, each party has 2 shares of the output sharing (m_(b_r), r_s, r_h).
fn multiply_bits_by_public_integers(
    c: Node,
    b: Node,
    integer_owner_id: u64,
    prf_keys: Node,
) -> Result<Node> {
    let g = c.get_graph();

    let party_s_id = integer_owner_id;
    let party_h_id = (integer_owner_id + 1) % (PARTIES as u64);
    let party_r_id = PARTIES as u64 - party_s_id - party_h_id;

    // Extract second and third PRF keys, which are known to parties R, S and parties S, H, respectively.
    let key_rs = prf_keys.tuple_get(party_s_id)?;
    let key_sh = prf_keys.tuple_get(party_h_id)?;

    // Compute (b_s XOR b_h) * c
    let bs_xor_bh = b.tuple_get(party_s_id)?.add(b.tuple_get(party_h_id)?)?;
    let bs_xor_bh_times_c = c.mixed_multiply(bs_xor_bh)?;

    // 1. Parties R and S sample pseudo-random r_s.
    let rs = g.prf(key_rs, 0, bs_xor_bh_times_c.get_type()?)?;
    // 2. Parties S and H sample pseudo-random r_h.
    let rh = g.prf(key_sh.clone(), 0, bs_xor_bh_times_c.get_type()?)?;
    // 3. Party S  creates two messages using its shares b_s, b_h of b:
    //    - m_0 = (b_s XOR b_h) * c - r_s - r_h;
    //    - m_1 = NOT(b_s XOR b_h) * c - r_s - r_h = c - (b_s XOR b_h) * c - r_s - r_h.
    //    Note that (m_(b_r), r_s, r_h) is the sharing of b * c
    let m0 = bs_xor_bh_times_c
        .subtract(rs.clone())?
        .subtract(rh.clone())?;
    let m1 = c
        .subtract(bs_xor_bh_times_c)?
        .subtract(rs.clone())?
        .subtract(rh.clone())?;
    // 4. Since parties R and H know b_r and party S knows m_0, m_1,
    //    they run the oblivious transfer protocol, which result in party R receiving m_(b_r).
    let br = b.tuple_get(party_r_id)?;
    let m_br = g.custom_op(
        CustomOperation::new(ObliviousTransfer {
            sender_id: party_s_id,
            receiver_id: party_r_id,
        }),
        vec![m0, m1, br, key_sh],
    )?;
    // 5. Party R shares m_(b_r) with party H.
    let sent_m_br = m_br
        .nop()?
        .add_annotation(NodeAnnotation::Send(party_r_id, party_h_id))?;
    // As a result, each party has 2 shares of the output sharing (m_(b_r), r_s, r_h).
    let mut shares = vec![sent_m_br; 3];
    shares[party_s_id as usize] = rs;
    shares[party_h_id as usize] = rh;
    g.create_tuple(shares)
}

#[typetag::serde]
impl CustomOperationBody for MixedMultiplyMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if !(2..=3).contains(&argument_types.len()) {
            return Err(runtime_error!(
                "MixedMultiplyMPC should have either 2 or 3 inputs"
            ));
        }
        let g = context.create_graph()?;
        let t_a = argument_types[0].clone();
        let t_b = argument_types[1].clone();
        let a = g.input(t_a.clone())?;
        let b = g.input(t_b.clone())?;

        // If an input is private, i.e. a tuple of 3 elements (a0, a1, a2), then
        // the parties can access the following elements:
        // 1st party -> a0, a1;
        // 2nd party -> a1, a2;
        // 3rd party -> a2, a0.
        match (t_a, t_b) {
            (Type::Tuple(v_a), Type::Tuple(v_b)) => {
                // Both integers and bits are private.
                // In this case, parties engage in two instances of the above protocol to obtain [a0 * b] and [(a1+a2) * b].
                // In the former instance, the integer owner is party 0, which knows the share a0.
                // In the latter one, the integer owner is party 1, which knows both shares a1 and a2; thus, it can compute their sum.
                // The final step is to sum shares [a0 * b] and [(a1+a2) * b], which yields [a0 * b + (a1+a2) * b] = [a * b]
                check_private_tuple(v_a)?;
                check_private_tuple(v_b)?;
                if argument_types.len() != 3 {
                    return Err(runtime_error!("MixedMultiply with two private inputs should be provided a tuple of PRF keys"));
                }
                let prf_type = argument_types[2].clone();
                let prf_keys = g.input(prf_type)?;

                let a0 = a.tuple_get(0)?;
                let a1_plus_a2 = a.tuple_get(1)?.add(a.tuple_get(2)?)?;

                let a0_times_b =
                    multiply_bits_by_public_integers(a0, b.clone(), 0, prf_keys.clone())?;
                let a1_plus_a2_times_b =
                    multiply_bits_by_public_integers(a1_plus_a2, b, 1, prf_keys)?;

                let mut ab_shares = vec![];
                for i in 0..PARTIES as u64 {
                    let ab_i = a0_times_b
                        .tuple_get(i)?
                        .add(a1_plus_a2_times_b.tuple_get(i)?)?;
                    ab_shares.push(ab_i);
                }
                g.create_tuple(ab_shares)?.set_as_output()?;
            }
            (Type::Tuple(v_a), Type::Array(_, _) | Type::Scalar(_)) => {
                // Integers are private, bits are public.
                // This means that output should be either private integers or zeros that can be computed locally.
                check_private_tuple(v_a)?;
                mixed_product(a, b, g.clone(), Operation::MixedMultiply, false)?;
            }
            (Type::Array(_, _) | Type::Scalar(_), Type::Tuple(v1)) => {
                // Integers are public, bits are private.
                // In this case, bits are multiplied by public integers using the above protocol with party 1 having a role of the integer owner.
                check_private_tuple(v1)?;
                if argument_types.len() != 3 {
                    return Err(runtime_error!(
                        "MixedMultiply with private bits should be provided a tuple of PRF keys"
                    ));
                }
                let prf_type = argument_types[2].clone();
                let prf_keys = g.input(prf_type)?;

                // All parties know a including party 1
                let o = multiply_bits_by_public_integers(a, b, 1, prf_keys)?;

                o.set_as_output()?;
            }
            (Type::Array(_, _) | Type::Scalar(_), Type::Array(_, _) | Type::Scalar(_)) => {
                // Both integers and bits are public.
                // No MPC-specific compilation is needed.
                let o = a.mixed_multiply(b)?;
                o.set_as_output()?;
            }
            _ => {
                panic!("Inconsistency with type checker");
            }
        }
        g.finalize()
    }

    fn get_name(&self) -> String {
        "MixedMultiplyMPC".to_owned()
    }
}

pub(super) fn general_multiply_mpc(
    input_a: Node,
    input_b: Node,
    op: Operation,
    prf_keys_mul: Option<Node>,
    reshare_needed: bool,
) -> Result<Node> {
    let custom_op = match op {
        Operation::Multiply => CustomOperation::new(MultiplyMPC {}),
        Operation::MixedMultiply => CustomOperation::new(MixedMultiplyMPC {}),
        Operation::Dot => CustomOperation::new(DotMPC {}),
        Operation::Matmul => CustomOperation::new(MatmulMPC {}),
        Operation::Gemm(transpose_a, transpose_b) => CustomOperation::new(GemmMPC {
            transpose_a,
            transpose_b,
        }),
        _ => {
            return Err(runtime_error!("Should not be here"));
        }
    };
    let g = input_a.get_graph();
    // Public inputs must be arrays and shared inputs must be tuples
    if (is_array_shared(&input_a)? || op == Operation::MixedMultiply) && is_array_shared(&input_b)?
    {
        // If both inputs are private, the MPC protocol requires invoking PRFs.
        // Thus, PRF keys must be provided.
        let keys = match prf_keys_mul {
            Some(ref k) => k.clone(),
            None => {
                return Err(runtime_error!("Propagation of annotations failed"));
            }
        };
        if op == Operation::MixedMultiply {
            g.custom_op(custom_op, vec![input_a, input_b, keys])
        } else {
            let product_node = g.custom_op(custom_op, vec![input_a, input_b])?;
            if reshare_needed {
                reshare(&product_node, &keys)
            } else {
                Ok(product_node)
            }
        }
    } else {
        g.custom_op(custom_op, vec![input_a, input_b])
    }
}

// If we add multiplication nodes manually, reshare their results by default.
pub(super) fn multiply_mpc(a: Node, b: Node, prf_keys: Node, reshare_needed: bool) -> Result<Node> {
    general_multiply_mpc(a, b, Operation::Multiply, Some(prf_keys), reshare_needed)
}

pub(super) fn mixed_multiply_mpc(a: Node, b: Node, prf_keys: Node) -> Result<Node> {
    general_multiply_mpc(a, b, Operation::MixedMultiply, Some(prf_keys), true)
}

pub(super) fn gemm_mpc(
    a: Node,
    b: Node,
    transpose_a: bool,
    transpose_b: bool,
    prf_keys: Node,
    reshare_needed: bool,
) -> Result<Node> {
    general_multiply_mpc(
        a,
        b,
        Operation::Gemm(transpose_a, transpose_b),
        Some(prf_keys),
        reshare_needed,
    )
}

pub(super) fn add_mpc(a: Node, b: Node) -> Result<Node> {
    a.get_graph()
        .custom_op(CustomOperation::new(AddMPC {}), vec![a, b])
}

pub(super) fn subtract_mpc(a: Node, b: Node) -> Result<Node> {
    a.get_graph()
        .custom_op(CustomOperation::new(SubtractMPC {}), vec![a, b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::subtract_vectors_u128;
    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::{
        array_type, scalar_type, tuple_type, ArrayShape, ScalarType, BIT, INT128, INT32, UINT128,
        UINT32,
    };
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;
    use crate::inline::inline_ops::{inline_operations, InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{generate_prf_key_triple, prepare_for_mpc_evaluation, IOStatus};
    use crate::mpc::mpc_equivalence_class::{generate_equivalence_class, EquivalenceClasses};
    use std::sync::Arc;

    fn prepare_arithmetic_context(
        input_party_map: Vec<IOStatus>,
        output_parties: Vec<IOStatus>,
        op: Operation,
        st: ScalarType,
        dims: Vec<ArrayShape>,
    ) -> Result<Context> {
        let c = simple_context(|g| {
            let mut types = vec![];
            if op == Operation::MixedMultiply {
                types.push(array_type(dims[0].clone(), st));
                types.push(array_type(dims[1].clone(), BIT));
                types.push(array_type(dims[2].clone(), BIT));
            } else {
                for shape in dims {
                    types.push(array_type(shape, st));
                }
            }
            let i1 = g.input(types[0].clone())?;
            i1.set_name("Input 1")?;
            let i2 = g.input(types[1].clone())?;
            i2.set_name("Input 2")?;
            match op {
                Operation::Add => {
                    let a1 = i1.add(i2)?;
                    a1.add(g.input(types[2].clone())?)
                }
                Operation::Subtract => {
                    let a1 = i1.subtract(i2)?;
                    a1.subtract(g.input(types[2].clone())?)
                }
                Operation::Multiply
                | Operation::Dot
                | Operation::Matmul
                | Operation::MixedMultiply
                | Operation::Gemm(_, _) => {
                    let a1 = bilinear_product(i1, i2, op.clone())?;
                    bilinear_product(a1, g.input(types[2].clone())?, op)
                }
                _ => {
                    panic!("Shouldn't be here");
                }
            }
        })?;
        let inline_config = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };

        let mpc_c = prepare_for_mpc_evaluation(
            &c,
            vec![input_party_map],
            vec![output_parties],
            inline_config,
        )?
        .get_context();
        // Check names
        let mpc_graph = mpc_c.get_main_graph()?;
        assert!(mpc_c.retrieve_node(mpc_graph.clone(), "Input 1").is_ok());
        assert!(mpc_c.retrieve_node(mpc_graph, "Input 2").is_ok());
        Ok(mpc_c)
    }

    fn prepare_arithmetic_input(
        input: Vec<Vec<u128>>,
        input_status: Vec<IOStatus>,
        st: Vec<ScalarType>,
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
                let first_share = subtract_vectors_u128(&input[i], &threes, st[i].get_modulus())?;
                v.push(Value::from_flattened_array(&first_share, st[i])?);
                for j in 1..PARTIES {
                    let share = vec![j; n];
                    v.push(Value::from_flattened_array(&share, st[i])?);
                }
                res.push(Value::from_vector(v));
            } else {
                res.push(Value::from_flattened_array(&input[i], st[i])?);
            }
        }
        Ok(res)
    }

    fn check_arithmetic_output(
        mpc_graph: Graph,
        inputs: Vec<Value>,
        expected: Vec<u128>,
        st: ScalarType,
        dims: ArrayShape,
        output_parties: Vec<IOStatus>,
    ) -> Result<()> {
        let output = random_evaluate(mpc_graph, inputs)?;
        let output_type = array_type(dims.clone(), st);

        let out = if output_parties.is_empty() {
            // check that mpc_output is a sharing of plain_output
            assert!(output.check_type(tuple_type(vec![output_type.clone(); PARTIES]))?);
            // add up shared values.
            output.access_vector(|v| {
                let flat_dims: u64 = dims.iter().product();
                let mut res = vec![0; flat_dims as usize];
                for val in v {
                    let arr = val.to_flattened_array_u128(output_type.clone())?;
                    for i in 0..flat_dims {
                        res[i as usize] = u128::wrapping_add(res[i as usize], arr[i as usize]);
                    }
                }
                Ok(res)
            })?
        } else {
            assert!(output.check_type(output_type.clone())?);
            output.to_flattened_array_u128(output_type)?
        };
        let out = if let Some(m) = st.get_modulus() {
            out.iter().map(|x| x % m).collect::<Vec<_>>()
        } else {
            out
        };
        assert_eq!(out, expected);
        Ok(())
    }

    fn helper_add_subtract(
        op: Operation,
        st: ScalarType,
        input: Vec<Vec<u128>>,
        expected: Vec<u128>,
        dims_in: Vec<ArrayShape>,
        dims_out: ArrayShape,
    ) -> Result<()> {
        let helper = |op: Operation,
                      st: ScalarType,
                      input: Vec<Vec<u128>>,
                      expected: Vec<u128>,
                      input_status: Vec<IOStatus>,
                      output_parties: Vec<IOStatus>|
         -> Result<()> {
            let mpc_context = prepare_arithmetic_context(
                input_status.clone(),
                output_parties.clone(),
                op,
                st,
                dims_in.clone(),
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            let inputs = prepare_arithmetic_input(input, input_status, vec![st; 3])?;

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
            st,
            input.clone(),
            expected.clone(),
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            vec![IOStatus::Party(0)],
        )?;
        helper(
            op.clone(),
            st,
            input.clone(),
            expected.clone(),
            vec![IOStatus::Public, IOStatus::Party(0), IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )?;
        helper(
            op.clone(),
            st,
            input.clone(),
            expected.clone(),
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Party(0)],
            vec![IOStatus::Party(0), IOStatus::Party(1)],
        )?;
        helper(
            op.clone(),
            st,
            input.clone(),
            expected.clone(),
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )?;
        helper(
            op.clone(),
            st,
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
        helper_add_subtract(
            Operation::Add,
            UINT128,
            vec![vec![2, 2], vec![0, 2], vec![u128::MAX - 2, u128::MAX - 5]],
            vec![u128::MAX, u128::MAX - 1],
            vec![vec![2]; 3],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Add,
            INT128,
            vec![vec![1, 2], vec![i128::MAX as u128; 2], vec![1, 1]],
            vec![i128::MIN as u128 + 1, i128::MIN as u128 + 2],
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

        helper_add_subtract(
            Operation::Subtract,
            UINT128,
            vec![vec![u128::MAX, 10], vec![5], vec![4]],
            vec![u128::MAX - 9, 1],
            vec![vec![2], vec![1], vec![1]],
            vec![2],
        )
        .unwrap();
        helper_add_subtract(
            Operation::Subtract,
            INT128,
            vec![vec![10, 9], vec![4, 5], vec![7, 6]],
            vec![u128::MAX, u128::MAX - 1],
            vec![vec![2]; 3],
            vec![2],
        )
        .unwrap();
    }

    fn bilinear_product_helper(op: Operation, dims: ArrayShape, st: ScalarType) -> Result<()> {
        let helper = |input_status: Vec<IOStatus>, output_parties: Vec<IOStatus>| -> Result<()> {
            let mpc_context = prepare_arithmetic_context(
                input_status.clone(),
                output_parties.clone(),
                op.clone(),
                st,
                vec![dims.clone(); 3],
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            let flat_dims = dims.iter().product::<u64>() as u128;
            let inputs = prepare_arithmetic_input(
                vec![
                    (2..2 + flat_dims).collect(),
                    (4..4 + flat_dims).collect(),
                    (6..6 + flat_dims).collect(),
                ],
                input_status,
                vec![st; 3],
            )?;

            let expected = match op.clone() {
                Operation::Multiply => vec![48, 105],
                Operation::Dot => vec![138, 161],
                Operation::Matmul | Operation::Gemm(_, _) => vec![404, 461, 716, 817],
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
        bilinear_product_helper(Operation::Multiply, vec![2], INT32).unwrap();
        bilinear_product_helper(Operation::Multiply, vec![2], INT128).unwrap();
    }

    #[test]
    fn test_dot() {
        bilinear_product_helper(Operation::Dot, vec![2], INT32).unwrap();
        bilinear_product_helper(Operation::Dot, vec![2], INT128).unwrap();
    }

    #[test]
    fn test_matmul() {
        bilinear_product_helper(Operation::Matmul, vec![2, 2], INT32).unwrap();
        bilinear_product_helper(Operation::Matmul, vec![2, 2], INT128).unwrap();
    }

    #[test]
    fn test_gemm() {
        bilinear_product_helper(Operation::Gemm(false, false), vec![2, 2], INT32).unwrap();
        bilinear_product_helper(Operation::Gemm(false, false), vec![2, 2], INT128).unwrap();
    }

    fn mixed_multiply_helper(st: ScalarType) -> Result<()> {
        let dims = vec![2, 2];
        let helper = |input_status: Vec<IOStatus>, output_parties: Vec<IOStatus>| -> Result<()> {
            let mpc_context = prepare_arithmetic_context(
                input_status.clone(),
                output_parties.clone(),
                Operation::MixedMultiply,
                st,
                vec![dims.clone(); 3],
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            let inputs = prepare_arithmetic_input(
                vec![vec![2, 3, 4, 5], vec![1, 1, 0, 1], vec![0, 1, 1, 1]],
                input_status,
                vec![st, BIT, BIT],
            )?;

            let expected = vec![0, 3, 0, 5];

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
        )?;
        helper(
            vec![IOStatus::Public, IOStatus::Party(0), IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )?;
        helper(
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Party(0)],
            vec![IOStatus::Party(0), IOStatus::Party(1)],
        )?;
        helper(
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )?;
        helper(
            vec![IOStatus::Public, IOStatus::Party(0), IOStatus::Public],
            vec![],
        )?;
        helper(
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![],
        )?;

        Ok(())
    }

    #[test]
    fn test_mixed_multiply_correctness() -> Result<()> {
        mixed_multiply_helper(INT32)?;
        mixed_multiply_helper(INT128)
    }

    #[test]
    fn test_mixed_multiply_communication() -> Result<()> {
        let c = simple_context(|g| {
            let input_type1 = tuple_type(vec![scalar_type(INT128); 3]);
            let input_type2 = tuple_type(vec![scalar_type(BIT); 3]);
            let i1 = g.input(input_type1)?;
            let i2 = g.input(input_type2)?;
            let prf_keys = {
                let keys_vec = generate_prf_key_triple(g.clone())?;
                g.create_tuple(keys_vec)?
            };
            g.custom_op(
                CustomOperation::new(MixedMultiplyMPC {}),
                vec![i1, i2, prf_keys],
            )
        })?;

        let instantiated_c = run_instantiation_pass(c)?.context;
        let inlined_c = inline_operations(
            &instantiated_c,
            InlineConfig {
                default_mode: InlineMode::Simple,
                ..Default::default()
            },
        )?
        .get_context();
        let result_class =
            generate_equivalence_class(&inlined_c, vec![vec![IOStatus::Shared, IOStatus::Shared]])?;

        let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
        let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);
        let share2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);
        let shared = EquivalenceClasses::Vector(vec![
            Arc::new(share1_02),
            Arc::new(share2_01),
            Arc::new(share0_12),
        ]);

        let main_graph = inlined_c.get_main_graph()?;
        let output_node_id = main_graph.get_output_node()?.get_id();

        // Output should be shared. TODO: test other nodes
        assert_eq!(*result_class.get(&(0, output_node_id)).unwrap(), shared);

        Ok(())
    }
}
