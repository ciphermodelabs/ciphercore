use crate::custom_ops::{run_instantiation_pass, ContextMappings, CustomOperation, MappedContext};
use crate::data_types::{array_type, scalar_type, tuple_type, Type, TypePointer, BIT, UINT64};
use crate::data_values::Value;
use crate::errors::Result;
use crate::evaluators::Evaluator;
use crate::graphs::{
    copy_node_name, create_context, Context, Graph, Node, NodeAnnotation, Operation,
};
use crate::inline::inline_ops::{inline_operations, InlineConfig};
use crate::mpc::mpc_apply_permutation::ApplyPermutationMPC;
use crate::mpc::mpc_arithmetic::{
    AddMPC, DotMPC, MatmulMPC, MixedMultiplyMPC, MultiplyMPC, SubtractMPC,
};
use crate::mpc::mpc_conversion::{A2BMPC, B2AMPC};
use crate::mpc::mpc_truncate::{TruncateMPC, TruncateMPC2K};
use crate::optimizer::optimize::optimize_context;

use std::collections::HashMap;
use std::collections::HashSet;

use super::mpc_arithmetic::GemmMPC;
use super::mpc_psi::JoinMPC;
use super::mpc_radix_sort::RadixSortMPC;

// We implement the ABY3 protocol, which has 3 parties involved
pub const PARTIES: usize = 3;

// Ownership status of input/output nodes
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IOStatus {
    Public,     // input/output is public / known to all the parties
    Party(u64), // input is privately owned by party i; output is revealed to party i
    Shared,     // input/output is shared / unknown to all the parties
}

// Bitsize of PRF keys
pub const KEY_LENGTH: u64 = 128;

/// Checks whether a private tuple value has the correct number of shares
pub(super) fn check_private_tuple(v: Vec<TypePointer>) -> Result<()> {
    if v.len() != PARTIES {
        return Err(runtime_error!(
            "Private tuple should have {} values, but {} provided",
            PARTIES,
            v.len()
        ));
    }
    let t = (*v[0]).clone();
    for coef in v.iter().skip(1) {
        if t != **coef {
            return Err(runtime_error!(
                "Private tuple should have value of the same type"
            ));
        }
    }
    Ok(())
}

fn is_one_node_private(nodes: &[Node], private_nodes: &HashSet<Node>) -> bool {
    for node in nodes {
        if private_nodes.contains(node) {
            return true;
        }
    }
    false
}

fn are_all_nodes_private(nodes: &[Node], private_nodes: &HashSet<Node>) -> bool {
    for node in nodes {
        if !private_nodes.contains(node) {
            return false;
        }
    }
    true
}

/// Generate random share of a given node: (node + alpha_0, alpha_1, alpha_2),
/// where alpha_i = PRF(k_i, 0) - PRF(k_(i+1 % 3), 0).
/// If no node is given, generate random shares of zero.
/// The input node is given as a pair of the node and the index of the party that wants to share this node.
/// The iv of any PRF call is set to zero here, but it will be changed to a unique
/// number by `uniquify_prf_id` when preparing an MPC graph for evaluation.
fn recursively_generate_node_shares(
    g: Graph,
    prf_keys: Vec<Node>,
    t: Type,
    node_to_share: Option<(Node, IOStatus)>,
) -> Result<Vec<Node>> {
    match t {
        Type::Scalar(_) | Type::Array(_, _) => {
            let mut random_shares = vec![];
            for key in prf_keys {
                let prf_i = g.prf(key, 0, t.clone())?;
                random_shares.push(prf_i);
            }
            let mut node_shares = vec![];
            for i in 0..PARTIES {
                let ip1 = (i + 1) % PARTIES;
                let alpha = g.subtract(random_shares[i].clone(), random_shares[ip1].clone())?;
                node_shares.push(alpha);
            }

            match node_to_share {
                Some((node, IOStatus::Party(id))) => {
                    node_shares[id as usize] = node_shares[id as usize].add(node)?;
                }
                Some((node, IOStatus::Public)) => {
                    node_shares[0] = node_shares[0].add(node)?;
                }
                Some((_, IOStatus::Shared)) => {
                    return Err(runtime_error!(
                        "Given node must belong to a party or be public"
                    ));
                }
                None => (),
            }
            Ok(node_shares)
        }
        Type::Tuple(types) => {
            let mut unpacked_node_shares = vec![vec![]; PARTIES];
            for (i, sub_t) in types.iter().enumerate() {
                let sub_node_to_share = match node_to_share.clone() {
                    Some((node, party_id)) => Some((node.tuple_get(i as u64)?, party_id)),
                    None => None,
                };
                let sub_node_shares = recursively_generate_node_shares(
                    g.clone(),
                    prf_keys.clone(),
                    (**sub_t).clone(),
                    sub_node_to_share,
                )?;
                for party_id in 0..PARTIES {
                    unpacked_node_shares[party_id].push(sub_node_shares[party_id].clone());
                }
            }
            let mut node_shares = vec![];
            for unpacked_share in unpacked_node_shares {
                node_shares.push(g.create_tuple(unpacked_share)?);
            }
            Ok(node_shares)
        }
        Type::Vector(length, element_type) => {
            let mut unpacked_node_shares = vec![vec![]; PARTIES];
            for i in 0..length {
                let sub_node_to_share = match node_to_share.clone() {
                    Some((node, party_id)) => {
                        let i_node =
                            g.constant(scalar_type(UINT64), Value::from_scalar(i, UINT64)?)?;
                        Some((node.vector_get(i_node)?, party_id))
                    }
                    None => None,
                };
                let sub_node_shares = recursively_generate_node_shares(
                    g.clone(),
                    prf_keys.clone(),
                    (*element_type).clone(),
                    sub_node_to_share,
                )?;
                for party_id in 0..PARTIES {
                    unpacked_node_shares[party_id].push(sub_node_shares[party_id].clone());
                }
            }
            let mut node_shares = vec![];
            for unpacked_share in unpacked_node_shares {
                node_shares.push(g.create_vector((*element_type).clone(), unpacked_share)?);
            }
            Ok(node_shares)
        }
        Type::NamedTuple(names_types) => {
            let mut unpacked_node_shares = vec![vec![]; PARTIES];
            for (name, sub_t) in &names_types {
                let sub_node_to_share = match node_to_share.clone() {
                    Some((node, party_id)) => {
                        Some((node.named_tuple_get((*name).clone())?, party_id))
                    }
                    None => None,
                };
                let sub_node_shares = recursively_generate_node_shares(
                    g.clone(),
                    prf_keys.clone(),
                    (**sub_t).clone(),
                    sub_node_to_share,
                )?;
                for party_id in 0..PARTIES {
                    unpacked_node_shares[party_id]
                        .push(((*name).clone(), sub_node_shares[party_id].clone()));
                }
            }
            let mut node_shares = vec![];
            for unpacked_share in unpacked_node_shares {
                node_shares.push(g.create_named_tuple(unpacked_share)?);
            }
            Ok(node_shares)
        }
    }
}

pub(crate) fn get_node_shares(
    g: Graph,
    prf_keys: Node,
    t: Type,
    node_to_share: Option<(Node, IOStatus)>,
) -> Result<Vec<Node>> {
    let mut prf_keys_vec = vec![];
    for i in 0..PARTIES {
        let key = g.tuple_get(prf_keys.clone(), i as u64)?;
        prf_keys_vec.push(key);
    }
    recursively_generate_node_shares(g, prf_keys_vec, t, node_to_share)
}

pub(super) fn get_zero_shares(g: Graph, prf_keys: Node, t: Type) -> Result<Vec<Node>> {
    get_node_shares(g, prf_keys, t, None)
}

/// Returns the hash set of the private nodes of the given graph,
/// a Boolean value indicating whether PRF keys should be used for multiplication,
/// a Boolean value indicating whether PRF keys should be used for B2A.
/// a Boolean value indicating whether PRF keys should be used for Truncate2k.
fn propagate_private_annotations(
    graph: Graph,
    is_input_private: Vec<bool>,
) -> Result<(HashSet<Node>, bool, bool, bool)> {
    let mut private_nodes: HashSet<Node> = HashSet::new();
    let mut use_prf_for_mul = false;
    let mut use_prf_for_b2a = false;
    let mut use_prf_for_truncate2k = false;
    let mut input_id = 0usize;
    for node in graph.get_nodes() {
        let op = node.get_operation();
        match op {
            Operation::Input(_) => {
                if is_input_private[input_id] {
                    private_nodes.insert(node);
                }
                input_id += 1;
            }
            Operation::Add
            | Operation::Subtract
            | Operation::Multiply
            | Operation::MixedMultiply
            | Operation::Dot
            | Operation::Matmul
            | Operation::Gemm(_, _)
            | Operation::Join(_, _)
            | Operation::A2B
            | Operation::B2A(_)
            | Operation::PermuteAxes(_)
            | Operation::ArrayToVector
            | Operation::TupleGet(_)
            | Operation::NamedTupleGet(_)
            | Operation::VectorToArray
            | Operation::GetSlice(_)
            | Operation::Reshape(_)
            | Operation::Sum(_)
            | Operation::Get(_)
            | Operation::CreateTuple
            | Operation::CreateNamedTuple(_)
            | Operation::CreateVector(_)
            | Operation::Stack(_)
            | Operation::ApplyPermutation(_)
            | Operation::Sort(_)
            | Operation::Concatenate(_)
            | Operation::Zip
            | Operation::Repeat(_) => {
                let dependencies = node.get_node_dependencies();
                if is_one_node_private(&dependencies, &private_nodes) {
                    private_nodes.insert(node.clone());
                    if matches!(op, Operation::Join(_, _)) {
                        use_prf_for_mul = true;
                    }
                }
                if ([
                    Operation::Multiply,
                    Operation::Dot,
                    Operation::Matmul,
                    Operation::A2B,
                ]
                .contains(&op)
                    || matches!(op, Operation::Gemm(_, _)))
                    && are_all_nodes_private(&dependencies, &private_nodes)
                {
                    use_prf_for_mul = true;
                }
                if matches!(op, Operation::B2A(_))
                    && are_all_nodes_private(&dependencies, &private_nodes)
                {
                    use_prf_for_mul = true;
                    use_prf_for_b2a = true;
                }
                if matches!(op, Operation::Sort(_)) && private_nodes.contains(&dependencies[0]) {
                    use_prf_for_mul = true;
                }
                if matches!(op, Operation::ApplyPermutation(_))
                    && private_nodes.contains(&dependencies[1])
                {
                    use_prf_for_mul = true;
                }
                if matches!(op, Operation::MixedMultiply)
                    && private_nodes.contains(&dependencies[1])
                {
                    use_prf_for_mul = true;
                }
            }
            Operation::Truncate(scale) => {
                let dependencies = node.get_node_dependencies();
                if is_one_node_private(&dependencies, &private_nodes) {
                    private_nodes.insert(node.clone());
                }

                if are_all_nodes_private(&dependencies, &private_nodes) {
                    use_prf_for_mul = true;
                    if scale.is_power_of_two() {
                        use_prf_for_truncate2k = true;
                    }
                }
            }
            Operation::Constant(_, _) | Operation::Zeros(_) | Operation::Ones(_) => {
                // Constants are always public
            }
            Operation::VectorGet => {
                let dependencies = node.get_node_dependencies();
                if private_nodes.contains(&dependencies[1]) {
                    return Err(runtime_error!("VectorGet can't have a private index"));
                }
                if private_nodes.contains(&dependencies[0]) {
                    private_nodes.insert(node.clone());
                }
            }
            _ => {
                return Err(runtime_error!(
                    "MPC compiler can't preprocess inputs of {}",
                    op
                ));
            }
        }
    }
    Ok((
        private_nodes,
        use_prf_for_mul,
        use_prf_for_b2a,
        use_prf_for_truncate2k,
    ))
}

pub(super) fn compile_to_mpc_graph(
    in_graph: Graph,
    is_input_private: Vec<bool>,
    out_context: Context,
    out_mapping: &mut ContextMappings,
) -> Result<Graph> {
    let out_graph = out_context.create_graph()?;

    let (private_nodes, use_prf_for_mul, use_prf_for_b2a, use_prf_for_truncate2k) =
        propagate_private_annotations(in_graph.clone(), is_input_private)?;
    // Input tuple of PRF keys for multiplication if needed
    // If created, these are the first input node of a graph
    let prf_keys_mul = if use_prf_for_mul {
        // PRF key type
        let key_t = array_type(vec![KEY_LENGTH], BIT);
        let key_inputs = vec![key_t; PARTIES];
        let keys_type = tuple_type(key_inputs);
        let node = out_graph.input(keys_type)?;
        node.add_annotation(NodeAnnotation::PRFMultiplication)?;
        Some(node)
    } else {
        None
    };
    // Input tuple of PRF keys for B2A if needed
    // If created, these are the second input node of a graph
    let prf_keys_b2a = if use_prf_for_b2a {
        // PRF key type
        let key_t = array_type(vec![KEY_LENGTH], BIT);
        let key_inputs = vec![key_t; PARTIES];
        let key_triple_type = tuple_type(key_inputs);
        let keys_type = tuple_type(vec![key_triple_type; 2]);
        let node = out_graph.input(keys_type)?;
        node.add_annotation(NodeAnnotation::PRFB2A)?;
        Some(node)
    } else {
        None
    };
    // Input tuple of PRF keys for Truncate if needed
    // If created, these are the second input node of a graph
    let prf_keys_truncate2k = if use_prf_for_truncate2k {
        // PRF key type
        let key_t = array_type(vec![KEY_LENGTH], BIT);
        let node = out_graph.input(key_t)?;
        node.add_annotation(NodeAnnotation::PRFTruncate)?;
        Some(node)
    } else {
        None
    };

    // This helper closure applies the operation (op) of a given node (node_to_be_private)
    // to given input nodes (node_dependencies).
    // If the node is public, the operation is applied as in the plaintext evaluator.
    // If the node is private, the helper extracts secret shares from input nodes
    // or promote public nodes to private and apply the operation on each share.
    // The following assumptions should be satisfied by the helper inputs:
    // - every input of a public node is a public node;
    // - a constant node is public;
    // - every private node has at least one private input node;
    // all the public inputs are promoted to private (except for VectorGet);
    // - a private VectorGet node has a private input vector node and a public index node.
    let apply_op = |node_to_be_private: Node,
                    op: Operation,
                    node_dependencies: Vec<Node>,
                    old_dependencies: Vec<Node>|
     -> Result<Node> {
        if !private_nodes.contains(&node_to_be_private) {
            return out_graph.add_node(node_dependencies, vec![], op);
        }
        if let Operation::Input(t) = op.clone() {
            let tuple_t = tuple_type(vec![t; PARTIES]);
            return out_graph.input(tuple_t);
        }
        let mut result_shares = vec![];
        for i in 0..PARTIES {
            let share = match op.clone() {
                Operation::VectorGet => vec![
                    out_graph.tuple_get(node_dependencies[0].clone(), i as u64)?,
                    node_dependencies[1].clone(),
                ],
                _ => {
                    let mut share_vec = vec![];
                    for (j, old_node) in old_dependencies.iter().enumerate() {
                        if private_nodes.contains(old_node) {
                            // if node is private, take the corresponding share
                            let new_node =
                                out_graph.tuple_get(node_dependencies[j].clone(), i as u64)?;
                            share_vec.push(new_node);
                        } else {
                            // if node is public, we promote it to private by splitting into (node, 0, 0)
                            // where 0 is a node containing zeros
                            if i == 0 {
                                share_vec.push(node_dependencies[j].clone())
                            } else {
                                let t = node_dependencies[j].get_type()?;
                                let new_node =
                                    out_graph.constant(t.clone(), Value::zero_of_type(t))?;
                                share_vec.push(new_node);
                            }
                        }
                    }
                    share_vec
                }
            };
            let result_share = out_graph.add_node(share, vec![], op.clone())?;
            result_shares.push(result_share);
        }
        out_graph.create_tuple(result_shares)
    };

    for node in in_graph.get_nodes() {
        let op = node.get_operation();
        let new_node = match op.clone() {
            Operation::Input(_) => apply_op(node.clone(), op, vec![], vec![])?,
            Operation::Add | Operation::Subtract => {
                let dependencies = node.get_node_dependencies();
                let input0 = dependencies[0].clone();
                let input1 = dependencies[1].clone();
                let new_input0 = out_mapping.get_node(input0);
                let new_input1 = out_mapping.get_node(input1);
                let custom_op = match op.clone() {
                    Operation::Add => CustomOperation::new(AddMPC {}),
                    Operation::Subtract => CustomOperation::new(SubtractMPC {}),
                    _ => panic!("Should not be here"),
                };
                out_graph.custom_op(custom_op, vec![new_input0.clone(), new_input1.clone()])?
            }
            Operation::Multiply | Operation::MixedMultiply | Operation::Dot | Operation::Matmul => {
                let dependencies = node.get_node_dependencies();
                let input0 = dependencies[0].clone();
                let input1 = dependencies[1].clone();
                let new_input0 = out_mapping.get_node(input0.clone());
                let new_input1 = out_mapping.get_node(input1.clone());
                let custom_op = match op.clone() {
                    Operation::Multiply => CustomOperation::new(MultiplyMPC {}),
                    Operation::MixedMultiply => CustomOperation::new(MixedMultiplyMPC {}),
                    Operation::Dot => CustomOperation::new(DotMPC {}),
                    Operation::Matmul => CustomOperation::new(MatmulMPC {}),
                    _ => panic!("Should not be here"),
                };

                if (private_nodes.contains(&input0) || op == Operation::MixedMultiply)
                    && private_nodes.contains(&input1)
                {
                    // If both inputs are private, the MPC protocol requires invoking PRFs.
                    // Thus, PRF keys must be provided.
                    let keys = match prf_keys_mul {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };
                    out_graph.custom_op(
                        custom_op,
                        vec![new_input0.clone(), new_input1.clone(), keys],
                    )?
                } else {
                    out_graph.custom_op(custom_op, vec![new_input0.clone(), new_input1.clone()])?
                }
            }
            Operation::Gemm(transpose_a, transpose_b) => {
                let dependencies = node.get_node_dependencies();
                let input0 = dependencies[0].clone();
                let input1 = dependencies[1].clone();
                let new_input0 = out_mapping.get_node(input0.clone());
                let new_input1 = out_mapping.get_node(input1.clone());
                let custom_op = CustomOperation::new(GemmMPC {
                    transpose_a,
                    transpose_b,
                });

                if (private_nodes.contains(&input0) || op == Operation::MixedMultiply)
                    && private_nodes.contains(&input1)
                {
                    // If both inputs are private, the MPC protocol requires invoking PRFs.
                    // Thus, PRF keys must be provided.
                    let keys = match prf_keys_mul {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };
                    out_graph.custom_op(
                        custom_op,
                        vec![new_input0.clone(), new_input1.clone(), keys],
                    )?
                } else {
                    out_graph.custom_op(custom_op, vec![new_input0.clone(), new_input1.clone()])?
                }
            }
            Operation::Join(join_t, headers) => {
                let dependencies = node.get_node_dependencies();
                let input0 = dependencies[0].clone();
                let input1 = dependencies[1].clone();
                let new_input0 = out_mapping.get_node(input0.clone());
                let new_input1 = out_mapping.get_node(input1.clone());
                let mut headers_vec = vec![];
                for headers_pair in headers {
                    headers_vec.push(headers_pair);
                }
                let custom_op = CustomOperation::new(JoinMPC {
                    join_t,
                    headers: headers_vec,
                });

                if private_nodes.contains(&node) {
                    // If one input set is private, MPC protocols requires invoking PRFs.
                    // Thus, PRF keys must be provided.
                    let keys = match prf_keys_mul {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };
                    out_graph.custom_op(
                        custom_op,
                        vec![new_input0.clone(), new_input1.clone(), keys],
                    )?
                } else {
                    out_graph.custom_op(custom_op, vec![new_input0.clone(), new_input1.clone()])?
                }
            }
            Operation::ApplyPermutation(inverse_permutation) => {
                let dependencies = node.get_node_dependencies();
                let input = dependencies[0].clone();
                let permutation = dependencies[1].clone();
                let new_input = out_mapping.get_node(input.clone());
                let new_permutation = out_mapping.get_node(permutation.clone());
                let custom_op = CustomOperation::new(ApplyPermutationMPC {
                    inverse_permutation,
                    reveal_output: false,
                });
                if private_nodes.contains(&permutation) {
                    // If the permutation is private, MPC protocols requires invoking PRFs.
                    // Thus, PRF keys must be provided.
                    let keys = match prf_keys_mul {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };
                    out_graph.custom_op(custom_op, vec![new_input, new_permutation, keys])?
                } else {
                    out_graph.custom_op(custom_op, vec![new_input, new_permutation])?
                }
            }
            Operation::Sort(key) => {
                let dependencies = node.get_node_dependencies();
                let mut mapped_dependencies = dependencies
                    .into_iter()
                    .map(|d| out_mapping.get_node(d))
                    .collect::<Vec<Node>>();
                let custom_op = CustomOperation::new(RadixSortMPC::new(key));
                if private_nodes.contains(&node) {
                    // If one input set is private, MPC protocols requires invoking PRFs.
                    // Thus, PRF keys must be provided.
                    let keys = match prf_keys_mul {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };
                    mapped_dependencies.push(keys);
                }
                out_graph.custom_op(custom_op, mapped_dependencies)?
            }
            Operation::Truncate(scale) => {
                let dependencies = node.get_node_dependencies();
                let input = dependencies[0].clone();
                let new_input = out_mapping.get_node(input.clone());
                let custom_op = if scale.is_power_of_two() {
                    let k = scale.trailing_zeros() as u64;
                    CustomOperation::new(TruncateMPC2K { k })
                } else {
                    CustomOperation::new(TruncateMPC { scale })
                };
                if private_nodes.contains(&input) {
                    // If input is private, the MPC protocol requires invoking PRFs.
                    // Thus, PRF keys must be provided.
                    let prf_mul_keys = match prf_keys_mul {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };

                    if scale.is_power_of_two() {
                        let prf_truncate_keys = match prf_keys_truncate2k {
                            Some(ref k) => k.clone(),
                            None => {
                                return Err(runtime_error!("Propagation of annotations failed"));
                            }
                        };

                        out_graph.custom_op(
                            custom_op,
                            vec![new_input.clone(), prf_mul_keys, prf_truncate_keys],
                        )?
                    } else {
                        out_graph.custom_op(custom_op, vec![new_input.clone(), prf_mul_keys])?
                    }
                } else {
                    out_graph.custom_op(custom_op, vec![new_input.clone()])?
                }
            }
            Operation::A2B => {
                let dependencies = node.get_node_dependencies();
                let input = dependencies[0].clone();
                let new_input = out_mapping.get_node(input.clone());
                let custom_op = CustomOperation::new(A2BMPC {});
                if private_nodes.contains(&input) {
                    // If input is private, the MPC protocol requires invoking PRFs.
                    // Thus, PRF keys must be provided.
                    let keys = match prf_keys_mul {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };
                    out_graph.custom_op(custom_op, vec![new_input.clone(), keys])?
                } else {
                    out_graph.custom_op(custom_op, vec![new_input.clone()])?
                }
            }
            Operation::B2A(st) => {
                let dependencies = node.get_node_dependencies();
                let input = dependencies[0].clone();
                let new_input = out_mapping.get_node(input.clone());
                let custom_op = CustomOperation::new(B2AMPC { st });
                if private_nodes.contains(&input) {
                    // If input is private, the MPC protocol requires invoking PRFs.
                    // Thus, PRF keys must be provided.
                    let keys_mul = match prf_keys_mul {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };
                    let keys_b2a = match prf_keys_b2a {
                        Some(ref k) => k.clone(),
                        None => {
                            return Err(runtime_error!("Propagation of annotations failed"));
                        }
                    };
                    out_graph.custom_op(custom_op, vec![new_input.clone(), keys_mul, keys_b2a])?
                } else {
                    out_graph.custom_op(custom_op, vec![new_input.clone()])?
                }
            }
            Operation::Constant(t, v) => out_graph.constant(t, v)?,
            Operation::Zeros(t) => out_graph.zeros(t)?,
            Operation::Ones(t) => out_graph.ones(t)?,
            Operation::PermuteAxes(_)
            | Operation::ArrayToVector
            | Operation::VectorToArray
            | Operation::TupleGet(_)
            | Operation::NamedTupleGet(_)
            | Operation::GetSlice(_)
            | Operation::Reshape(_)
            | Operation::Sum(_)
            | Operation::Get(_)
            | Operation::Repeat(_) => {
                let dependencies = node.get_node_dependencies();
                let input = dependencies[0].clone();
                let new_input = out_mapping.get_node(input.clone());
                apply_op(input, op, vec![new_input], dependencies)?
            }
            Operation::VectorGet => {
                let dependencies = node.get_node_dependencies();
                let vector = dependencies[0].clone();
                let index = dependencies[1].clone();
                let new_vector = out_mapping.get_node(vector.clone());
                let new_index = out_mapping.get_node(index.clone());

                apply_op(vector, op, vec![new_vector, new_index], vec![])?
            }
            Operation::CreateTuple
            | Operation::CreateNamedTuple(_)
            | Operation::CreateVector(_)
            | Operation::Stack(_)
            | Operation::Concatenate(_)
            | Operation::Zip => {
                let dependencies = node.get_node_dependencies();
                let new_dependencies: Vec<Node> = dependencies
                    .iter()
                    .map(|x| out_mapping.get_node((*x).clone()))
                    .collect();
                apply_op(node.clone(), op, new_dependencies, dependencies)?
            }
            _ => {
                return Err(runtime_error!(
                    "MPC compilation for {} not yet implemented",
                    op
                ));
            }
        };
        if private_nodes.contains(&node) {
            new_node.add_annotation(NodeAnnotation::Private)?;
        }
        out_mapping.insert_node(node, new_node);
    }
    let old_output_node = in_graph.get_output_node()?;
    let output_node = out_mapping.get_node(old_output_node);
    out_graph.set_output_node(output_node)?;
    out_graph.finalize()?;
    Ok(out_graph)
}

fn contains_node_annotation(g: Graph, annotation: NodeAnnotation) -> Result<bool> {
    let nodes = g.get_nodes();
    for node in nodes {
        let annotations = node.get_annotations()?;
        if annotations.contains(&annotation) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn share_node(g: Graph, node: Node, prf_keys: Node, status: IOStatus) -> Result<Node> {
    let mut outputs = vec![];
    let t = node.get_type()?;
    let node_shares = get_node_shares(g.clone(), prf_keys, t, Some((node, status)))?;
    // networking
    for (i, node_share) in node_shares.iter().enumerate().take(PARTIES) {
        let network_node = g.nop((*node_share).clone())?;
        let im1 = ((i + PARTIES - 1) % PARTIES) as u64;
        network_node.add_annotation(NodeAnnotation::Send(i as u64, im1))?;
        outputs.push(network_node);
    }
    g.create_tuple(outputs)
}

fn share_input(g: Graph, node: Node, t: Type, prf_keys: Node, status: IOStatus) -> Result<Node> {
    let plain_input = g.input(t)?;
    copy_node_name(node, plain_input.clone())?;
    share_node(g, plain_input, prf_keys, status)
}

/// Generates a triple of random PRF keys (k_0, k_1, k_2) such that k_i is generated by party i.
/// The keys are then distributed such that
/// the ith party has k_i and k_{i+1} (the index is taken modulo 3).
pub(super) fn generate_prf_key_triple(g: Graph) -> Result<Vec<Node>> {
    let key_t = array_type(vec![KEY_LENGTH], BIT);
    let mut triple = vec![];
    for party_id in 0..PARTIES {
        let key = g.random(key_t.clone())?;
        let key_sent = g.nop(key)?;
        let prev_party_id = (party_id + PARTIES - 1) % PARTIES;
        key_sent.add_annotation(NodeAnnotation::Send(party_id as u64, prev_party_id as u64))?;
        triple.push(key_sent);
    }
    Ok(triple)
}

fn share_all_inputs(
    in_graph: Graph,
    out_graph: Graph,
    input_party_map: Vec<IOStatus>,
    prf_keys: Node,
    is_prf_mul_key_needed: bool,
    is_prf_b2a_key_needed: bool,
    is_prf_truncate_key_needed: bool,
) -> Result<Vec<Node>> {
    let mut shared_inputs = if is_prf_mul_key_needed {
        vec![prf_keys.clone()]
    } else {
        vec![]
    };
    if is_prf_b2a_key_needed {
        // Create PRF keys for B2A.
        // These are 2 tuples ((k_00, k_01, k_02), (k_10, k_11, k_12)) where
        // k_00, k_01, k_02, k_10, k_11 should be known to party 0,
        // k_01, k_02, k_10, k_11, k_12 should be known to party 1,
        // all these keys should be known to party 2.
        // Party i generate keys k_0i and k_1i and send it to other parties.
        let prf_b2a_key = {
            let mut keys = vec![];
            for _ in 0..2 {
                let key_triple = generate_prf_key_triple(out_graph.clone())?;
                keys.push(key_triple);
            }
            // Stopping here will result in the following access pattern
            // k_00, k_01, k_10, k_11 should be known to party 0,
            // k_01, k_02, k_11, k_12 should be known to party 1,
            // k_00, k_02, k_10, k_12 should be known to party 2.
            // This means that party 0 should send k_10 to party 1,
            keys[1][0] = keys[1][0].nop()?;
            keys[1][0].add_annotation(NodeAnnotation::Send(0, 1))?;
            // party 1 should send k_01, k_11 to party 2
            keys[0][1] = keys[0][1].nop()?;
            keys[0][1].add_annotation(NodeAnnotation::Send(1, 2))?;
            keys[1][1] = keys[1][1].nop()?;
            keys[1][1].add_annotation(NodeAnnotation::Send(1, 2))?;
            // and party 2 should send k_02 to party 0.
            keys[0][2] = keys[0][2].nop()?;
            keys[0][2].add_annotation(NodeAnnotation::Send(2, 0))?;

            let key_triple0 = out_graph.create_tuple(keys[0].clone())?;
            let key_triple1 = out_graph.create_tuple(keys[1].clone())?;
            out_graph.create_tuple(vec![key_triple0, key_triple1])?
        };
        shared_inputs.push(prf_b2a_key);
    }
    if is_prf_truncate_key_needed {
        let key_t = array_type(vec![KEY_LENGTH], BIT);
        let prf_truncate_key = out_graph.random(key_t)?;
        shared_inputs.push(prf_truncate_key);
    }

    let mut input_id = 0usize;
    for node in in_graph.get_nodes() {
        if let Operation::Input(t) = node.get_operation() {
            let shared_input = match input_party_map[input_id] {
                IOStatus::Party(_) => share_input(
                    out_graph.clone(),
                    node.clone(),
                    t,
                    prf_keys.clone(),
                    input_party_map[input_id].clone(),
                )?,
                IOStatus::Shared => {
                    let new_node = out_graph.input(tuple_type(vec![t.clone(); PARTIES]))?;
                    copy_node_name(node.clone(), new_node.clone())?;
                    new_node
                }
                IOStatus::Public => {
                    let new_node = out_graph.input(t)?;
                    copy_node_name(node.clone(), new_node.clone())?;
                    new_node
                }
            };
            input_id += 1;
            shared_inputs.push(shared_input);
        }
    }
    Ok(shared_inputs)
}

pub(super) fn recursively_sum_shares(g: Graph, shares: Vec<Node>) -> Result<Node> {
    let t = shares[0].get_type()?;
    match t {
        Type::Scalar(_) | Type::Array(_, _) => {
            let mut res = shares[0].clone();
            for share in shares.iter().skip(1) {
                res = res.add(share.clone())?;
            }
            Ok(res)
        }
        Type::Tuple(types) => {
            let mut revealed_sub_nodes = vec![];
            for i in 0..types.len() as u64 {
                let mut sub_shares = vec![];
                for share in &shares {
                    let sub_share = share.tuple_get(i)?;
                    sub_shares.push(sub_share);
                }
                let revealed_sub_node = recursively_sum_shares(g.clone(), sub_shares)?;
                revealed_sub_nodes.push(revealed_sub_node);
            }
            g.create_tuple(revealed_sub_nodes)
        }
        Type::Vector(length, element_type) => {
            let mut revealed_sub_nodes = vec![];
            for i in 0..length {
                let i_node = g.constant(scalar_type(UINT64), Value::from_scalar(i, UINT64)?)?;
                let mut sub_shares = vec![];
                for share in &shares {
                    let sub_share = share.vector_get(i_node.clone())?;
                    sub_shares.push(sub_share);
                }
                let revealed_sub_node = recursively_sum_shares(g.clone(), sub_shares)?;
                revealed_sub_nodes.push(revealed_sub_node);
            }
            g.create_vector((*element_type).clone(), revealed_sub_nodes)
        }
        Type::NamedTuple(names_types) => {
            let mut revealed_sub_nodes = vec![];
            for (name, _) in names_types {
                let mut sub_shares = vec![];
                for share in &shares {
                    let sub_share = share.named_tuple_get(name.clone())?;
                    sub_shares.push(sub_share);
                }
                let revealed_sub_node = recursively_sum_shares(g.clone(), sub_shares)?;
                revealed_sub_nodes.push((name, revealed_sub_node));
            }
            g.create_named_tuple(revealed_sub_nodes)
        }
    }
}

/// Output parties ids must be in the range 0..PARTIES.
fn reveal_output(g: Graph, out_node: Node, output_parties: Vec<IOStatus>) -> Result<Node> {
    // If there are no parties obtaining revealed output, return output in the shared form
    if output_parties.is_empty() {
        return Ok(out_node);
    }
    // Extract output shares
    let mut shares = vec![];
    for i in 0..PARTIES as u64 {
        let share = out_node.tuple_get(i)?;
        shares.push(share);
    }
    if let IOStatus::Party(id) = output_parties[0] {
        let party_id = id as usize;
        let mut shares_to_reveal = shares.clone();
        // Networking to obtain missing shares
        let prev_party_id = (party_id + PARTIES - 1) % PARTIES;
        let missing_share = shares_to_reveal[prev_party_id].nop()?;
        shares_to_reveal[prev_party_id] = missing_share
            .add_annotation(NodeAnnotation::Send(prev_party_id as u64, party_id as u64))?;
        // Sum shares
        let revealed_node = recursively_sum_shares(g, shares_to_reveal)?;
        // If there are other parties waiting for a revealed value, send it to them
        let result_node = if output_parties.len() > 1 {
            let mut send_node = revealed_node;
            for i in 1..PARTIES {
                let party_to_send_id = (party_id + i) % PARTIES;
                if output_parties.contains(&IOStatus::Party(party_to_send_id as u64)) {
                    send_node = send_node.nop()?;
                    send_node.add_annotation(NodeAnnotation::Send(
                        party_id as u64,
                        party_to_send_id as u64,
                    ))?;
                }
            }
            // Output node can't have Send annotation
            send_node.nop()?
        } else {
            revealed_node
        };
        return Ok(result_node);
    }
    panic!("Shouldn't be here");
}

/// Compiles all the graphs of an already inlined context into graphs for secure computation and add it to another context.
/// Namely, every plaintext operation is replaced by a related MPC protocol from the ABY3 framework.
/// The given input-party map describes assigns every input to one of the following statuses:
/// - public,
/// - already shared,
/// - should be shared by certain party.
/// The `output_parties` argument contains ids of the parties (from 0..PARTIES) that obtain the revealed result of MPC computation.
fn compile_to_mpc_context(
    in_context: Context,
    input_party_map: Vec<Vec<IOStatus>>,
    output_parties: Vec<Vec<IOStatus>>,
    out_context: Context,
    out_mapping: &mut ContextMappings,
) -> Result<()> {
    in_context.check_finalized()?;

    for (i, graph) in in_context.get_graphs().iter().enumerate() {
        // compile the current graph to MPC
        let is_input_private: Vec<bool> = input_party_map[i]
            .iter()
            .map(|x| *x != IOStatus::Public)
            .collect();
        let computation_graph = compile_to_mpc_graph(
            graph.clone(),
            is_input_private.clone(),
            out_context.clone(),
            out_mapping,
        )?;

        let new_graph = out_context.create_graph()?;
        // Input tuple of PRF keys for zero sharing.
        let prf_keys = {
            let keys_vec = generate_prf_key_triple(new_graph.clone())?;
            new_graph.create_tuple(keys_vec)?
        };

        // input nodes that are followed by secret sharing
        let is_prf_mul_key_needed =
            contains_node_annotation(computation_graph.clone(), NodeAnnotation::PRFMultiplication)?;
        let is_prf_b2a_key_needed =
            contains_node_annotation(computation_graph.clone(), NodeAnnotation::PRFB2A)?;
        let is_prf_truncate_key_needed =
            contains_node_annotation(computation_graph.clone(), NodeAnnotation::PRFTruncate)?;
        let shared_input = share_all_inputs(
            graph.clone(),
            new_graph.clone(),
            input_party_map[i].clone(),
            prf_keys.clone(),
            is_prf_mul_key_needed,
            is_prf_b2a_key_needed,
            is_prf_truncate_key_needed,
        )?;

        // compute the MPC graph on the shared input
        let shared_result = new_graph.call(computation_graph.clone(), shared_input)?;
        // reveal the output to the given parties
        let is_output_private = {
            let out_node = computation_graph.get_output_node()?;
            let out_anno = out_node.get_annotations()?;
            out_anno.contains(&NodeAnnotation::Private)
        };
        let result = if is_output_private {
            reveal_output(new_graph.clone(), shared_result, output_parties[i].clone())?
        } else if output_parties[i].is_empty() {
            // if output is public and it should be secretly shared (no output parties), party 0 creates its secret sharing
            let node = share_node(
                new_graph.clone(),
                shared_result.clone(),
                prf_keys,
                IOStatus::Party(0),
            )?;
            node.add_annotation(NodeAnnotation::Private)?
        } else {
            shared_result
        };
        result.set_as_output()?;
        new_graph.finalize()?;
        out_mapping.insert_graph(graph.clone(), new_graph);
    }
    Ok(())
}

/// Compiles all the graphs of an already inlined context into graphs for secure computation.
/// Namely, every plaintext operation is replaced by a related MPC protocol from the ABY3 framework.
/// The given input-party map describes what statuses of inputs (public, shared or owned by a party).
/// The `output_parties` argument contains ids of the parties (from 0..PARTIES) that obtain the revealed result of MPC computation.
/// Input of PRF nodes is always zero. Thus, the resulting context is insecure to evaluate!
/// To guarantee security, unique PRF inputs are assigned later.
/// If private, the output of the main graph is always a tuple of 3 elements where the first element is known to the first party,
/// the second to the second one etc. Thus, the first tuple element can be either a share or a revealed value known to the first party.
fn compile_to_mpc(
    context: Context,
    input_party_map: Vec<Vec<IOStatus>>,
    output_parties: Vec<Vec<IOStatus>>,
) -> Result<MappedContext> {
    for sub_map in &input_party_map {
        for status in sub_map {
            if let IOStatus::Party(id) = *status {
                if id >= PARTIES as u64 {
                    return Err(runtime_error!("Input party should have a valid party ID"));
                }
            }
        }
    }
    for sub_parties in &output_parties {
        for status in sub_parties {
            if let IOStatus::Party(id) = *status {
                if id >= PARTIES as u64 {
                    return Err(runtime_error!("Output party should have a valid party ID"));
                }
            } else {
                return Err(runtime_error!(
                    "Output status should be a party id or shared"
                ));
            }
        }
    }
    let new_context = create_context()?;
    let mut context_map = ContextMappings::default();
    compile_to_mpc_context(
        context.clone(),
        input_party_map,
        output_parties,
        new_context.clone(),
        &mut context_map,
    )?;
    let old_main_graph = context.get_main_graph()?;
    let main_graph = context_map.get_graph(old_main_graph);
    new_context.set_main_graph(main_graph)?;
    new_context.finalize()?;
    let mut mapped_context = MappedContext::new(new_context);
    mapped_context.mappings = context_map;
    Ok(mapped_context)
}

/// Creates a new copy of an input context with PRF nodes containing globally unique inputs (iv's).
/// These global inputs are taken from the set {1,2,...,n} where n is the total number of PRF nodes.
pub fn uniquify_prf_id(context: Context) -> Result<Context> {
    let new_context = create_context()?;
    let mut context_map = ContextMappings::default();
    let graphs = context.get_graphs();
    let mut prf_id = 0;
    for graph in graphs {
        let out_graph = new_context.create_graph()?;
        let nodes = graph.get_nodes();
        for node in nodes {
            let op = node.get_operation();
            let op = if op.is_prf_operation() {
                prf_id += 1;
                op.update_prf_id(prf_id)?
            } else {
                op
            };
            let node_dependencies = node.get_node_dependencies();
            let new_node_dependencies: Vec<Node> = node_dependencies
                .iter()
                .map(|x| context_map.get_node((*x).clone()))
                .collect();
            let graph_dependencies = node.get_graph_dependencies();
            let new_graph_dependencies: Vec<Graph> = graph_dependencies
                .iter()
                .map(|x| context_map.get_graph((*x).clone()))
                .collect();
            let new_node = out_graph.add_node(new_node_dependencies, new_graph_dependencies, op)?;
            let annotations = node.get_annotations()?;
            for anno in annotations {
                new_node.add_annotation(anno)?;
            }
            copy_node_name(node.clone(), new_node.clone())?;
            context_map.insert_node(node, new_node);
        }
        let output_node = graph.get_output_node()?;
        let new_output_node = context_map.get_node(output_node);
        out_graph.set_output_node(new_output_node)?;
        out_graph.finalize()?;
        context_map.insert_graph(graph, out_graph.clone());
    }
    let old_main_graph = context.get_main_graph()?;
    let main_graph = context_map.get_graph(old_main_graph);
    new_context.set_main_graph(main_graph)?;
    new_context.finalize()?;
    Ok(new_context)
}

/// Converts a given inlined context to its counterpart that operates on MPC shares and is ready for evaluation.
/// It includes a call to the MPC compiler, the custom operation instantiation and inlining with a given configuration.
/// After inlining this function provides a unique input to every PRF node.
/// The resulting context preserves only the names of input nodes.
pub fn prepare_for_mpc_evaluation(
    context: Context,
    input_party_map: Vec<Vec<IOStatus>>,
    output_parties: Vec<Vec<IOStatus>>,
    inline_config: InlineConfig,
) -> Result<Context> {
    let mpc_context = compile_to_mpc(context, input_party_map, output_parties)?.get_context();
    let instantiated_context = run_instantiation_pass(mpc_context)?.get_context();
    let inlined_context = inline_operations(instantiated_context, inline_config)?;
    uniquify_prf_id(inlined_context)
}

fn print_stats(graph: Graph) -> Result<()> {
    let mut cnt = HashMap::<String, u64>::new();
    for node in graph.get_nodes() {
        let op_name = format!("{}", node.get_operation());
        *cnt.entry(op_name).or_insert(0) += 1;
    }
    let mut entries: Vec<(String, u64)> = cnt.iter().map(|e| (e.0.clone(), *e.1)).collect();
    entries.sort_by_key(|e| -(e.1 as i64));
    eprintln!("-------Stats--------");
    eprintln!("Total ops: {}", graph.get_nodes().len());
    for e in entries {
        eprintln!("{}\t{}", e.0, e.1);
    }
    Ok(())
}

pub fn prepare_context<E>(
    context: Context,
    inline_config: InlineConfig,
    evaluator: E,
    print_unoptimized_stats: bool,
) -> Result<Context>
where
    E: Evaluator + Sized,
{
    eprintln!("Instantiating...");
    let context2 = run_instantiation_pass(context)?.get_context();
    eprintln!("Inlining...");
    let context3 = inline_operations(context2, inline_config)?;
    if print_unoptimized_stats {
        print_stats(context3.get_main_graph()?)?;
    }
    eprintln!("Optimizing...");
    optimize_context(context3, evaluator)
}

/// Takes raw context (no inlining, etc.), and runs the whole pipeline (instantiation+inlining+MPC) on it,
/// to prepare to be used in runtime.
pub fn compile_context<T, E>(
    context: Context,
    input_parties: Vec<IOStatus>,
    output_parties: Vec<IOStatus>,
    inline_config: InlineConfig,
    get_evaluator: T,
) -> Result<Context>
where
    T: Fn() -> Result<E>,
    E: Evaluator + Sized,
{
    let evaluator0 = get_evaluator()?;
    let context4 = prepare_context(context, inline_config.clone(), evaluator0, true)?;
    print_stats(context4.get_main_graph()?)?;
    let mut number_of_inputs = 0;
    for node in context4.get_main_graph()?.get_nodes() {
        if node.get_operation().is_input() {
            number_of_inputs += 1;
        }
    }
    if input_parties.len() != number_of_inputs {
        return Err(runtime_error!(
            "Invalid number of input parties: {} expected, but {} found",
            number_of_inputs,
            input_parties.len()
        ));
    }
    eprintln!("input_parties = {input_parties:?}");
    eprintln!("output_parties = {output_parties:?}");
    let compiled_context0 = prepare_for_mpc_evaluation(
        context4,
        vec![input_parties],
        vec![output_parties],
        inline_config,
    )?;
    print_stats(compiled_context0.get_main_graph()?)?;

    let evaluator1 = get_evaluator()?;
    let compiled_context = optimize_context(compiled_context0, evaluator1)?;
    print_stats(compiled_context.get_main_graph()?)?;
    Ok(compiled_context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::{
        array_type, get_types_vector, named_tuple_type, scalar_type, tuple_type, vector_type, BIT,
        INT32, INT64, UINT64, UINT8, VOID_TYPE,
    };
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::evaluators::simple_evaluator::evaluate_add_subtract_multiply;
    use crate::graphs::util::simple_context;
    use crate::graphs::SliceElement::{Ellipsis, SubArray};
    use crate::inline::inline_ops::{inline_operations, InlineConfig, InlineMode};
    use crate::random::PRNG;

    use std::collections::HashMap;

    #[test]
    fn test_malformed() {
        || -> Result<()> {
            let c = create_context()?;
            assert!(compile_to_mpc(
                c.clone(),
                vec![vec![IOStatus::Public]],
                vec![vec![IOStatus::Party(0)]]
            )
            .is_err());
            let g = c.create_graph()?;
            g.input(scalar_type(BIT))?.set_as_output()?;
            g.finalize()?;
            c.set_main_graph(g)?;
            c.finalize()?;
            assert!(compile_to_mpc(
                c.clone(),
                vec![vec![IOStatus::Party(3)]],
                vec![vec![IOStatus::Party(0)]]
            )
            .is_err());
            assert!(compile_to_mpc(
                c.clone(),
                vec![vec![IOStatus::Public]],
                vec![vec![IOStatus::Party(5)]]
            )
            .is_err());
            assert!(compile_to_mpc(
                c.clone(),
                vec![vec![IOStatus::Public]],
                vec![vec![IOStatus::Shared]]
            )
            .is_err());
            Ok(())
        }()
        .unwrap();
    }

    fn reveal_private_value(value: Value, t: Type) -> Result<Value> {
        let shares = value.to_vector()?;
        if matches!(t.clone(), Type::Array(_, _) | Type::Scalar(_)) {
            let mut res = Value::zero_of_type(t.clone());
            for share in shares {
                res = evaluate_add_subtract_multiply(
                    t.clone(),
                    res.clone(),
                    t.clone(),
                    share,
                    Operation::Add,
                    t.clone(),
                )?;
            }
            return Ok(res);
        }

        let vector_types = get_types_vector(t.clone())?;
        let mut shares_vec = vec![];
        for i in 0..PARTIES {
            shares_vec.push(shares[i].to_vector()?);
        }
        let mut res_vec = vec![];
        for i in 0..vector_types.len() {
            let mut tuple_vec = vec![];
            for j in 0..PARTIES {
                tuple_vec.push(shares_vec[j][i].clone());
            }
            let tuple = Value::from_vector(tuple_vec);
            res_vec.push(reveal_private_value(tuple, (*vector_types[i]).clone())?);
        }
        Ok(Value::from_vector(res_vec))
    }

    #[test]
    fn test_input() {
        let seed = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F";
        let mut prng = PRNG::new(Some(*seed)).unwrap();
        let mut helper =
            |t: Type, input_status: IOStatus, output_parties: Vec<IOStatus>| -> Result<()> {
                let c = simple_context(|g| g.input(t.clone()))?;
                let mpc_mapped_context = compile_to_mpc(
                    c,
                    vec![vec![input_status.clone()]],
                    vec![output_parties.clone()],
                )?;
                let mpc_context = mpc_mapped_context.get_context();
                let mpc_graph = mpc_context.get_main_graph()?;
                let mut inputs = vec![];
                if input_status == IOStatus::Shared {
                    let tuple_t = tuple_type(vec![t.clone(); PARTIES]);
                    inputs.push(prng.get_random_value(tuple_t.clone())?);
                } else {
                    inputs.push(prng.get_random_value(t.clone())?);
                }
                let output = random_evaluate(mpc_graph.clone(), inputs.clone())?;

                let mpc_computation_graph = mpc_context.get_graphs()[0].clone();

                let computation_output_node = mpc_computation_graph.get_output_node()?;
                let computation_output_annotations = computation_output_node.get_annotations()?;
                if input_status != IOStatus::Public {
                    let expected = if input_status == IOStatus::Shared {
                        reveal_private_value(inputs[0].clone(), t.clone())?
                    } else {
                        inputs[0].clone()
                    };
                    // check that output is a sharing of expected
                    if output_parties.is_empty() {
                        let revealed_output = reveal_private_value(output.clone(), t.clone())?;
                        assert!(output.check_type(tuple_type(vec![t.clone(); PARTIES]))?);
                        assert_eq!(revealed_output, expected);
                    } else {
                        // check that output is of the right type
                        assert!(output.check_type(t.clone())?);
                        assert_eq!(output, expected.clone());
                    }
                    assert!(computation_output_annotations.contains(&NodeAnnotation::Private));
                } else {
                    // public input must be shared
                    if output_parties.is_empty() {
                        let revealed_output = reveal_private_value(output.clone(), t.clone())?;
                        assert!(output.check_type(tuple_type(vec![t.clone(); PARTIES]))?);
                        assert_eq!(revealed_output, inputs[0]);
                        // check that the final output is private (since it's shared)
                        let output_annotations = mpc_graph.get_output_node()?.get_annotations()?;
                        assert!(output_annotations.contains(&NodeAnnotation::Private));
                    } else {
                        assert_eq!(output, inputs[0]);
                    }
                    // computation output should be public on public inputs
                    assert!(!computation_output_annotations.contains(&NodeAnnotation::Private));
                }
                Ok(())
            };

        helper(
            array_type(vec![2, 2], INT64),
            IOStatus::Party(0),
            vec![IOStatus::Party(1)],
        )
        .unwrap();
        helper(
            array_type(vec![2, 2], INT64),
            IOStatus::Public,
            vec![IOStatus::Party(0)],
        )
        .unwrap();
        helper(
            scalar_type(UINT64),
            IOStatus::Party(1),
            vec![IOStatus::Party(1), IOStatus::Party(2)],
        )
        .unwrap();
        helper(
            scalar_type(UINT64),
            IOStatus::Public,
            vec![IOStatus::Party(0)],
        )
        .unwrap();
        helper(
            scalar_type(UINT64),
            IOStatus::Shared,
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )
        .unwrap();
        helper(scalar_type(UINT64), IOStatus::Shared, vec![]).unwrap();
        helper(scalar_type(UINT64), IOStatus::Public, vec![]).unwrap();
    }

    fn prepare_private_value(value: Value, t: Type) -> Result<Vec<Value>> {
        // private shares of value are generated as
        // value = (value + 2, -1, -1)
        if let Type::Scalar(st) | Type::Array(_, st) = t.clone() {
            let mut res = vec![];
            let zero = Value::zero_of_type(t.clone());
            let one = Value::from_scalar(1, st.clone())?;
            let two = Value::from_scalar(2, st.clone())?;
            for i in 0..PARTIES {
                let (add_sub, l_value, r_value) = match i {
                    0 => (Operation::Add, value.clone(), two.clone()),
                    1 => (Operation::Subtract, zero.clone(), one.clone()),
                    2 => (Operation::Subtract, zero.clone(), one.clone()),
                    _ => panic!("More than 3 parties are not supported"),
                };
                let share = evaluate_add_subtract_multiply(
                    t.clone(),
                    l_value,
                    scalar_type(st.clone()),
                    r_value,
                    add_sub,
                    t.clone(),
                )?;
                res.push(share);
            }
            return Ok(res);
        }

        let vector_types = get_types_vector(t.clone())?;
        let mut shares = vec![vec![]; PARTIES];
        value.access_vector(|vector_values| {
            for i in 0..vector_values.len() {
                let tuple_i =
                    prepare_private_value(vector_values[i].clone(), (*vector_types[i]).clone())?;
                for j in 0..PARTIES {
                    shares[j].push(tuple_i[j].clone())
                }
            }
            Ok(())
        })?;
        let mut res = vec![];
        for share in shares {
            res.push(Value::from_vector(share));
        }
        Ok(res)
    }

    fn prepare_value(value: Value, t: Type, is_input_private: bool) -> Result<Value> {
        if is_input_private {
            let tuple = prepare_private_value(value, t)?;
            return Ok(Value::from_vector(tuple));
        }
        Ok(value)
    }

    fn prepare_input(
        input_types: Vec<Type>,
        is_input_shared: Vec<bool>,
    ) -> Result<(Vec<Value>, Vec<Value>)> {
        let seed: [u8; 16] = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut prng = PRNG::new(Some(seed))?;
        let mut plain_inputs = vec![];
        let mut mpc_inputs = vec![];
        for i in 0..input_types.len() {
            let random_value = prng.get_random_value(input_types[i].clone())?;
            plain_inputs.push(random_value.clone());
            mpc_inputs.push(prepare_value(
                random_value,
                input_types[i].clone(),
                is_input_shared[i].clone(),
            )?);
        }
        Ok((plain_inputs, mpc_inputs))
    }

    fn check_output(
        plain_graph: Graph,
        mpc_graph: Graph,
        plain_inputs: Vec<Value>,
        mpc_inputs: Vec<Value>,
        output_parties: Vec<IOStatus>,
        t: Type,
    ) -> Result<()> {
        let plain_output = random_evaluate(plain_graph.clone(), plain_inputs)?;
        let mpc_output = random_evaluate(mpc_graph.clone(), mpc_inputs)?;

        if output_parties.is_empty() {
            // check that mpc_output is a sharing of plain_output
            assert!(mpc_output.check_type(tuple_type(vec![t.clone(); PARTIES]))?);
            let value_revealed = reveal_private_value(mpc_output.clone(), t.clone())?;
            assert_eq!(value_revealed, plain_output);
        } else {
            assert!(mpc_output.check_type(t.clone())?);
            assert_eq!(mpc_output, plain_output);
        }

        Ok(())
    }

    fn helper_one_input(
        input_types: Vec<Type>,
        op: Operation,
        input_party_map: Vec<IOStatus>,
        output_parties: Vec<IOStatus>,
    ) -> Result<()> {
        let c = simple_context(|g| {
            let mut input_nodes = vec![];
            for i in 0..input_types.len() {
                let input_node = g.input(input_types[i].clone())?;
                input_node.set_name(&format!("Input {}", i))?;
                input_nodes.push(input_node);
            }
            let o = if op != Operation::VectorGet {
                g.add_node(input_nodes, vec![], op)?
            } else {
                input_nodes[0].vector_get(g.zeros(scalar_type(UINT64))?)?
            };
            o.set_name("Plaintext operation")?;
            Ok(o)
        })?;
        let g = c.get_main_graph()?;
        let output_type = g.get_output_node()?.get_type()?;

        let inline_config = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };
        let mpc_c = prepare_for_mpc_evaluation(
            c.clone(),
            vec![input_party_map.clone()],
            vec![output_parties.clone()],
            inline_config,
        )?;
        let mpc_graph = mpc_c.get_main_graph()?;
        // Check names
        let mpc_node_result = mpc_c.retrieve_node(mpc_graph.clone(), "Plaintext operation");
        assert!(mpc_node_result.is_err());
        for i in 0..input_types.len() {
            let node_name = format!("Input {}", i);
            let new_input_node = mpc_c.retrieve_node(mpc_graph.clone(), &node_name);
            assert!(new_input_node.is_ok());
        }

        let is_input_shared = input_party_map
            .iter()
            .map(|x| *x == IOStatus::Shared)
            .collect();
        let (plain_inputs, mpc_inputs) = prepare_input(input_types.clone(), is_input_shared)?;

        check_output(
            g,
            mpc_graph,
            plain_inputs,
            mpc_inputs,
            output_parties,
            output_type,
        )?;
        Ok(())
    }

    fn test_helper_one_input(input_type: Type, op: Operation) -> Result<()> {
        helper_one_input(
            vec![input_type.clone()],
            op.clone(),
            vec![IOStatus::Party(0)],
            vec![IOStatus::Party(0)],
        )?;
        helper_one_input(
            vec![input_type.clone()],
            op.clone(),
            vec![IOStatus::Party(0)],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
        )?;
        helper_one_input(
            vec![input_type.clone()],
            op.clone(),
            vec![IOStatus::Public],
            vec![IOStatus::Party(1)],
        )?;
        helper_one_input(
            vec![input_type.clone()],
            op.clone(),
            vec![IOStatus::Shared],
            vec![IOStatus::Party(0), IOStatus::Party(1)],
        )?;
        helper_one_input(
            vec![input_type.clone()],
            op.clone(),
            vec![IOStatus::Party(0)],
            vec![],
        )?;
        helper_one_input(vec![input_type], op, vec![IOStatus::Public], vec![])?;
        Ok(())
    }

    #[test]
    fn test_permute_axes() {
        test_helper_one_input(
            array_type(vec![4, 2, 3], INT32),
            Operation::PermuteAxes(vec![2, 0, 1]),
        )
        .unwrap();
    }

    #[test]
    fn test_array_to_vector() {
        test_helper_one_input(array_type(vec![3, 1], UINT8), Operation::ArrayToVector).unwrap();
    }

    #[test]
    fn test_vector_to_array() {
        test_helper_one_input(
            vector_type(10, array_type(vec![4, 3], INT32)),
            Operation::VectorToArray,
        )
        .unwrap();
    }

    #[test]
    fn test_vector_get() {
        test_helper_one_input(
            vector_type(10, array_type(vec![4, 3], INT32)),
            Operation::VectorGet,
        )
        .unwrap();
    }

    #[test]
    fn test_get_slice() {
        test_helper_one_input(
            array_type(vec![10, 128], INT32),
            Operation::GetSlice(vec![Ellipsis, SubArray(None, Some(-1), None)]),
        )
        .unwrap();
    }

    #[test]
    fn test_reshape() {
        test_helper_one_input(
            array_type(vec![10, 128], INT32),
            Operation::Reshape(array_type(vec![20, 64], INT32)),
        )
        .unwrap();
    }

    #[test]
    fn test_tuple_get() {
        let t = array_type(vec![10, 128], INT32);
        test_helper_one_input(
            tuple_type(vec![t.clone(), scalar_type(UINT64), t]),
            Operation::TupleGet(1),
        )
        .unwrap();
    }

    #[test]
    fn test_named_tuple_get() {
        let t = array_type(vec![10, 128], INT32);
        test_helper_one_input(
            named_tuple_type(vec![
                ("a".to_owned(), t.clone()),
                ("b".to_owned(), scalar_type(UINT64)),
                ("c".to_owned(), t),
            ]),
            Operation::NamedTupleGet("b".to_string()),
        )
        .unwrap();
    }

    #[test]
    fn test_sum() {
        test_helper_one_input(
            array_type(vec![10, 5, 12], INT32),
            Operation::Sum(vec![0, 1]),
        )
        .unwrap();
    }

    #[test]
    fn test_get() {
        test_helper_one_input(array_type(vec![10, 128], INT32), Operation::Get(vec![5, 4]))
            .unwrap();
    }

    #[test]
    fn test_repeat() {
        test_helper_one_input(array_type(vec![10, 128], INT32), Operation::Repeat(10)).unwrap();
    }

    fn helper_create_ops(
        input_types: Vec<Type>,
        op: Operation,
        input_party_map: Vec<IOStatus>,
        output_parties: Vec<IOStatus>,
        include_constant: bool,
    ) -> Result<()> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let mut input_nodes = vec![];
        for i in 0..input_types.len() {
            let input_node = g.input(input_types[i].clone())?;
            input_node.set_name(&format!("Input {}", i))?;
            input_nodes.push(input_node);
        }
        let resolved_op = if include_constant {
            input_nodes.push(g.constant(
                input_types[0].clone(),
                Value::zero_of_type(input_types[0].clone()),
            )?);
            match op {
                Operation::CreateNamedTuple(mut names) => {
                    names.push("const".to_owned());
                    Operation::CreateNamedTuple(names)
                }
                Operation::Stack(outer_shape) => {
                    let mut pr = 1;
                    for x in &outer_shape {
                        pr *= *x;
                    }
                    Operation::Stack(vec![pr + 1])
                }
                _ => op,
            }
        } else {
            op
        };
        let o = g.add_node(input_nodes, vec![], resolved_op)?;
        o.set_name("Plaintext operation")?;
        let output_type = o.get_type()?;
        g.set_output_node(o.clone())?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;

        let inline_config = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };
        let mpc_c = prepare_for_mpc_evaluation(
            c.clone(),
            vec![input_party_map.clone()],
            vec![output_parties.clone()],
            inline_config,
        )?;
        let mpc_graph = mpc_c.get_main_graph()?;
        // Check names
        let mpc_node_result = mpc_c.retrieve_node(mpc_graph.clone(), "Plaintext operation");
        assert!(mpc_node_result.is_err());
        for i in 0..input_types.len() {
            let node_name = format!("Input {}", i);
            let new_input_node = mpc_c.retrieve_node(mpc_graph.clone(), &node_name);
            assert!(new_input_node.is_ok());
        }

        let is_input_shared = input_party_map
            .iter()
            .map(|x| *x == IOStatus::Shared)
            .collect();
        let (plain_inputs, mpc_inputs) = prepare_input(input_types.clone(), is_input_shared)?;

        check_output(
            g,
            mpc_graph,
            plain_inputs,
            mpc_inputs,
            output_parties,
            output_type,
        )?;
        Ok(())
    }

    fn test_helper_create_ops(input_types: Vec<Type>, op: Operation) -> Result<()> {
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            vec![IOStatus::Party(0)],
            true,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            true,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            vec![IOStatus::Party(0)],
            true,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            true,
        )?;

        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Party(0), IOStatus::Public, IOStatus::Party(1)],
            vec![IOStatus::Party(0)],
            true,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Party(0), IOStatus::Public, IOStatus::Party(1)],
            vec![IOStatus::Party(0)],
            true,
        )?;

        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![IOStatus::Party(0)],
            false,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![IOStatus::Party(0)],
            false,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            false,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Shared, IOStatus::Shared, IOStatus::Party(0)],
            vec![IOStatus::Party(0), IOStatus::Party(1)],
            true,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Shared, IOStatus::Shared, IOStatus::Party(0)],
            vec![],
            true,
        )?;
        helper_create_ops(
            input_types.clone(),
            op.clone(),
            vec![IOStatus::Public, IOStatus::Public, IOStatus::Public],
            vec![],
            true,
        )?;
        Ok(())
    }

    #[test]
    fn test_create_tuple() {
        let t = array_type(vec![10, 128], INT32);
        test_helper_create_ops(
            vec![t.clone(), scalar_type(UINT64), t.clone()],
            Operation::CreateTuple,
        )
        .unwrap();
        test_helper_create_ops(vec![t, VOID_TYPE], Operation::CreateTuple).unwrap();
    }

    #[test]
    fn test_create_named_tuple() {
        let t = array_type(vec![10, 128], INT32);
        test_helper_create_ops(
            vec![t.clone(), scalar_type(UINT64), t.clone()],
            Operation::CreateNamedTuple(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]),
        )
        .unwrap();
        test_helper_create_ops(
            vec![t, VOID_TYPE],
            Operation::CreateNamedTuple(vec!["a".to_owned(), "b".to_owned()]),
        )
        .unwrap();
    }

    #[test]
    fn test_create_vector() {
        let t = array_type(vec![10, 128], INT32);
        test_helper_create_ops(vec![t.clone(); 3], Operation::CreateVector(t)).unwrap();
    }

    #[test]
    fn test_zip() {
        let t = vector_type(10, array_type(vec![4, 3], INT32));
        test_helper_create_ops(vec![t.clone(); 3], Operation::Zip).unwrap();
    }
    #[test]
    fn test_stack() {
        let t = array_type(vec![10, 128], INT32);
        test_helper_create_ops(vec![t.clone(); 3], Operation::Stack(vec![3])).unwrap();
    }
    #[test]
    fn test_concatenate() {
        let t1 = array_type(vec![10, 1, 10], INT32);
        let t2 = array_type(vec![10, 2, 10], INT32);
        let t3 = array_type(vec![10, 3, 10], INT32);
        test_helper_create_ops(vec![t1, t2, t3], Operation::Concatenate(1)).unwrap();
    }

    // Checks that every PRF node of a context has a unique input.
    fn check_prf_id(context: Context) -> Result<()> {
        let mut iv_node_map: HashMap<u64, Node> = HashMap::new();
        let graphs = context.get_graphs();
        for graph in graphs {
            let nodes = graph.get_nodes();
            for node in nodes {
                let iv = match node.get_operation() {
                    Operation::PRF(iv, _) => iv,
                    Operation::PermutationFromPRF(iv, _) => iv,
                    _ => continue,
                };
                if let Some(other_node) = iv_node_map.get(&iv) {
                    if *other_node != node {
                        return Err(runtime_error!("PRF node with non-unique iv"));
                    }
                } else {
                    iv_node_map.insert(iv, node);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_prf_id() {
        || -> Result<()> {
            let c = create_context()?;
            let g1 = c.create_graph()?;
            {
                let i = g1.input(scalar_type(UINT64))?;
                let o = i.a2b()?;
                g1.set_output_node(o)?;
                g1.finalize()?;
            }
            let g2 = c.create_graph()?;
            {
                let i = g2.input(scalar_type(UINT64))?;
                let o = i.a2b()?;
                g2.set_output_node(o)?;
                g2.finalize()?;
            }

            c.set_main_graph(g2)?;
            c.finalize()?;

            let mpc_c = compile_to_mpc(
                c,
                vec![vec![IOStatus::Party(0)], vec![IOStatus::Party(1)]],
                vec![vec![IOStatus::Party(1)], vec![IOStatus::Party(2)]],
            )?
            .get_context();
            let instantiated_context = run_instantiation_pass(mpc_c)?.get_context();
            assert!(check_prf_id(instantiated_context.clone()).is_err());
            let inlined_context = inline_operations(
                instantiated_context.clone(),
                InlineConfig {
                    default_mode: InlineMode::Simple,
                    ..Default::default()
                },
            )?;
            assert!(check_prf_id(inlined_context.clone()).is_err());

            let validated_instantiated_context = uniquify_prf_id(instantiated_context)?;
            assert!(check_prf_id(validated_instantiated_context).is_ok());
            let validated_inlined_context = uniquify_prf_id(inlined_context)?;
            assert!(check_prf_id(validated_inlined_context).is_ok());
            Ok(())
        }()
        .unwrap()
    }

    #[test]
    fn test_prf_ids_for_permutation_from_prf() -> Result<()> {
        let c = create_context()?;
        let g1 = c.create_graph()?;
        {
            let k = g1.random(array_type(vec![128], BIT))?;
            g1.permutation_from_prf(k, 0, 10)?.set_as_output()?;
            g1.finalize()?;
        }
        let g2 = c.create_graph()?;
        {
            let k = g2.random(array_type(vec![128], BIT))?;
            g2.prf(k, 0, scalar_type(UINT64))?.set_as_output()?;
            g2.finalize()?;
        }
        let g3 = c.create_graph()?;
        {
            let k = g3.random(array_type(vec![128], BIT))?;
            g3.permutation_from_prf(k, 0, 11)?.set_as_output()?;
            g3.finalize()?;
        }
        let g4 = c.create_graph()?;
        {
            let k = g4.random(array_type(vec![128], BIT))?;
            g4.prf(k, 0, scalar_type(INT32))?.set_as_output()?;
            g4.finalize()?;
        }

        c.set_main_graph(g2)?;
        c.finalize()?;
        assert!(check_prf_id(c.clone()).is_err());

        let c = uniquify_prf_id(c)?;
        assert!(check_prf_id(c).is_ok());
        Ok(())
    }
}
