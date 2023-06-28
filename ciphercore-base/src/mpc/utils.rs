use std::ops::Not;

use serde::{Deserialize, Serialize};

use crate::bytes::{add_vectors_u128, subtract_vectors_u128};
use crate::custom_ops::{run_instantiation_pass, ContextMappings, CustomOperationBody};
use crate::data_types::{array_type, scalar_size_in_bytes, ScalarType, Type, BIT, UINT8};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, NodeAnnotation};
use crate::inline::inline_common::DepthOptimizationLevel;
use crate::inline::inline_ops::{inline_operations, InlineConfig, InlineMode};
use crate::random::PRNG;

use super::mpc_compiler::{compile_to_mpc_graph, KEY_LENGTH, PARTIES};

// Computes the oblivious transfer (OT) protocol that has the following input and output.
//
// Given two input values i_0 and i_1 known to a sender party, and a selection bit b known to two other parties,
// OT returns i_b to a receiver party and nothing to other parties.
// The receiver doesn't obtain any information about i_(1-b).
// The sender doesn't learn b.
// The helper (not a sender or receiver) doesn't learn i_0 or i_1.
//
// 4 input types should provided:
// - the first two should be equal arrays or scalars,
// - the third one should be a binary array or scalar,
// - the fourth one is a PRF key type known only to the sender and helper.
// The operation is performed elementwise applying broadcasting if necessary.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct ObliviousTransfer {
    pub sender_id: u64,
    pub receiver_id: u64, // The helper ID is defined automatically
}

#[typetag::serde]
impl CustomOperationBody for ObliviousTransfer {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() != 4 {
            return Err(runtime_error!(
                "Oblivious transport should have 4 input types"
            ));
        }
        if argument_types[0] != argument_types[1] {
            return Err(runtime_error!("First two input types should be equal"));
        }
        let bit_type = argument_types[2].clone();
        if bit_type.get_scalar_type() != BIT {
            return Err(runtime_error!(
                "Bit type should be a binary array or scalar"
            ));
        }
        let key_type = argument_types[3].clone();
        if key_type != array_type(vec![KEY_LENGTH], BIT) {
            return Err(runtime_error!(
                "Key type should be a binary array of length {}",
                KEY_LENGTH
            ));
        }
        if self.sender_id >= PARTIES as u64 {
            return Err(runtime_error!("Sender ID is incorrect"));
        }
        if self.receiver_id >= PARTIES as u64 {
            return Err(runtime_error!("Receiver ID is incorrect"));
        }
        if self.sender_id == self.receiver_id {
            return Err(runtime_error!(
                "Receiver ID should be different from the sender id"
            ));
        }

        let g = context.create_graph()?;

        // Input values known to the sender
        let input_type = argument_types[0].clone();
        let i0 = g.input(input_type.clone())?;
        let i1 = g.input(input_type.clone())?;

        // Selection bit known to the receiver and helper
        let b = g.input(bit_type)?;

        // PRF key known to the sender and helper.
        // It can be the corresponding PRF key for multiplication.
        let prf_key = g.input(key_type)?;

        // The sender and helper generate two random masks for the input values
        let r0 = g.prf(prf_key.clone(), 0, input_type.clone())?;
        let r1 = g.prf(prf_key, 0, input_type.clone())?;

        // The sender masks the input values and send them to the receiver
        let masked_i0 = i0
            .add(r0.clone())?
            .nop()?
            .add_annotation(NodeAnnotation::Send(self.sender_id, self.receiver_id))?;
        let masked_i1 = i1
            .add(r1.clone())?
            .nop()?
            .add_annotation(NodeAnnotation::Send(self.sender_id, self.receiver_id))?;

        // The helper selects r_b
        let rb = {
            let diff = r1.subtract(r0.clone())?;
            let diff_by_bit = if input_type.get_scalar_type() == BIT {
                diff.multiply(b.clone())?
            } else {
                diff.mixed_multiply(b.clone())?
            };
            diff_by_bit.add(r0)?
        };
        // The helper sends r_b to the receiver
        let helper_id = PARTIES as u64 - self.sender_id - self.receiver_id;
        let sent_rb = rb
            .nop()?
            .add_annotation(NodeAnnotation::Send(helper_id, self.receiver_id))?;

        // The receiver selects masked i_b
        let masked_ib = masked_i1
            .subtract(masked_i0.clone())?
            .mixed_multiply(b)?
            .add(masked_i0)?;

        // The receiver unmask i_b
        let ib = masked_ib.subtract(sent_rb)?;

        ib.set_as_output()?;
        g.finalize()
    }

    fn get_name(&self) -> String {
        format!(
            "OT(sender:{},receiver:{})",
            self.sender_id, self.receiver_id
        )
    }
}

/// Utility function for preparing secret-shared data.
///
/// TODO: in the future, we need to make secret-sharing more generic.
///
/// # Arguments
///
/// * `prng` - PRNG object (for randomness).
/// * `data` - array to secret-share.
/// * `scalar_type` - scalar type to use.
///
/// # Returns
///
/// Vector of shares.
pub fn share_vector<T: TryInto<u128> + Not<Output = T> + TryInto<u8> + Copy>(
    prng: &mut PRNG,
    data: &[T],
    scalar_type: ScalarType,
) -> Result<Vec<Value>> {
    let n = data.len();
    let n_bytes = n * scalar_size_in_bytes(scalar_type) as usize;

    // first share (r0) is pseudo-random
    let r0_bytes = prng.get_random_bytes(n_bytes)?;
    let r0 = Value::from_bytes(r0_bytes)
        .to_flattened_array_u128(array_type(vec![n as u64], scalar_type))?;
    // second share (r1) is pseudo-random
    let r1_bytes = prng.get_random_bytes(n_bytes)?;
    let r1 = Value::from_bytes(r1_bytes)
        .to_flattened_array_u128(array_type(vec![n as u64], scalar_type))?;
    // third share (r2) is r2 = data - (r0 + r1)
    let r0r1 = add_vectors_u128(&r0, &r1, scalar_type.get_modulus())?;
    let data_u128 = Value::from_flattened_array(data, scalar_type)?
        .to_flattened_array_u128(array_type(vec![n as u64], scalar_type))?;
    let r2 = subtract_vectors_u128(&data_u128, &r0r1, scalar_type.get_modulus())?;

    let shares = vec![
        Value::from_flattened_array(&r0, scalar_type)?,
        Value::from_flattened_array(&r1, scalar_type)?,
        Value::from_flattened_array(&r2, scalar_type)?,
    ];

    let mut garbage = vec![];
    for _ in 0..3 {
        garbage.push(Value::from_flattened_array(
            &prng.get_random_bytes(n_bytes)?,
            UINT8,
        )?);
    }

    // convert the shares to a value
    Ok(vec![
        Value::from_vector(vec![
            shares[0].clone(),
            shares[1].clone(),
            garbage[2].clone(),
        ]),
        Value::from_vector(vec![
            garbage[0].clone(),
            shares[1].clone(),
            shares[2].clone(),
        ]),
        Value::from_vector(vec![
            shares[0].clone(),
            garbage[1].clone(),
            shares[2].clone(),
        ]),
    ])
}

/// Selects elements of node 0 if bits of node b are zero and elements of node 1 otherwise.
///
/// Broadcasting is applied
///
/// # Arguments
///
/// - `a0` - node 0 containing an array
/// - `a1` - node 1 containing an array of the same type as `a`
/// - `b` - node containing selection bits
///
/// # Returns
///
/// Node containing an array with selected elements of `a0` or `a1`
pub fn select_node(b: Node, a1: Node, a0: Node) -> Result<Node> {
    let dif = a1.subtract(a0.clone())?;
    if dif.get_type()?.get_scalar_type() == BIT {
        dif.multiply(b)?.add(a0)
    } else {
        dif.mixed_multiply(b)?.add(a0)
    }
}

pub(super) fn convert_main_graph_to_mpc(
    in_context: Context,
    out_context: Context,
    is_input_private: Vec<bool>,
) -> Result<Graph> {
    let instantiated_context = run_instantiation_pass(in_context)?.get_context();
    let inlined_context = inline_operations(
        &instantiated_context,
        InlineConfig {
            default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            ..Default::default()
        },
    )?
    .get_context();

    let mut context_map = ContextMappings::default();

    // Compile to MPC
    let main_g_inlined = inlined_context.get_main_graph()?;
    let main_mpc_g = compile_to_mpc_graph(
        main_g_inlined,
        is_input_private,
        out_context,
        &mut context_map,
    )?;
    Ok(main_mpc_g)
}

pub(super) fn get_column(named_tuple_shares: &[Node], header: String) -> Result<Node> {
    if named_tuple_shares.len() == PARTIES {
        let mut shares = vec![];
        for share in named_tuple_shares {
            shares.push(share.named_tuple_get(header.clone())?);
        }
        named_tuple_shares[0].get_graph().create_tuple(shares)
    } else if named_tuple_shares.len() == 1 {
        named_tuple_shares[0].named_tuple_get(header)
    } else {
        Err(runtime_error!(
            "Wrong number of shares {}",
            named_tuple_shares.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        custom_ops::{run_instantiation_pass, CustomOperation},
        data_types::{INT32, UINT32},
        evaluators::random_evaluate,
        graphs::util::simple_context,
        inline::inline_ops::{inline_operations, InlineConfig, InlineMode},
        mpc::{
            mpc_compiler::IOStatus,
            mpc_equivalence_class::{generate_equivalence_class, EquivalenceClasses},
        },
    };

    #[test]
    fn test_simple_share() {
        || -> Result<()> {
            let data = vec![12, 34, 56];
            let mut prng = PRNG::new(None)?;
            let shares = share_vector(&mut prng, &data, UINT32)?;
            let shares0 = shares[0].to_vector()?;
            let shares1 = shares[1].to_vector()?;
            let shares2 = shares[2].to_vector()?;
            assert_eq!(shares0[0], shares2[0]);
            assert_eq!(shares0[1], shares1[1]);
            assert_eq!(shares1[2], shares2[2]);
            let t = array_type(vec![3], UINT32);
            let v0 = shares0[0].to_flattened_array_u128(t.clone())?;
            let v1 = shares1[1].to_flattened_array_u128(t.clone())?;
            let v2 = shares2[2].to_flattened_array_u128(t)?;
            let new_data = add_vectors_u128(
                &add_vectors_u128(&v0, &v1, UINT32.get_modulus())?,
                &v2,
                UINT32.get_modulus(),
            )?;
            assert_eq!(new_data, data);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_oblivious_transfer() {
        // test correct inputs
        let roles_helper = |sender_id: u64, receiver_id: u64| -> Result<()> {
            let helper_id = PARTIES as u64 - sender_id - receiver_id;
            let c = simple_context(|g| {
                let input_type = array_type(vec![2], INT32);
                let bit_type = array_type(vec![2], BIT);

                let i0 = g.input(input_type.clone())?;
                let i1 = g.input(input_type)?;

                // Generate a selecting bit known by parties 0 and 2
                let b = g
                    .input(bit_type)?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(receiver_id, helper_id))?;

                // Generate a PRF key known to the sender and helper
                let key_t = array_type(vec![KEY_LENGTH], BIT);
                let key = g
                    .random(key_t)?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(helper_id, sender_id))?;

                // Run the OT protocol with party 0 being a receiver and party 1 being a sender.
                g.custom_op(
                    CustomOperation::new(ObliviousTransfer {
                        sender_id,
                        receiver_id,
                    }),
                    vec![i0, i1, b, key],
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

            let result_class = generate_equivalence_class(
                &inlined_c,
                vec![vec![
                    IOStatus::Party(1),
                    IOStatus::Party(1),
                    IOStatus::Party(0),
                ]],
            )?;

            let private_class = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);
            // data shared by the sender and helper
            let share_r_sh =
                EquivalenceClasses::Atomic(vec![vec![receiver_id], vec![sender_id, helper_id]]);
            // data shared by the receiver and helper
            let share_s_rh =
                EquivalenceClasses::Atomic(vec![vec![sender_id], vec![receiver_id, helper_id]]);
            // data shared by the receiver and sender
            let share_h_rs =
                EquivalenceClasses::Atomic(vec![vec![helper_id], vec![receiver_id, sender_id]]);

            // both inputs should be known only to the sender
            assert_eq!(*result_class.get(&(0, 0)).unwrap(), private_class);
            assert_eq!(*result_class.get(&(0, 1)).unwrap(), private_class);
            // b must be known to the receiver and helper
            assert_eq!(*result_class.get(&(0, 2)).unwrap(), private_class);
            assert_eq!(*result_class.get(&(0, 3)).unwrap(), share_s_rh);
            // PRF key shared by the sender and helper
            assert_eq!(*result_class.get(&(0, 4)).unwrap(), private_class);
            assert_eq!(*result_class.get(&(0, 5)).unwrap(), share_r_sh);
            // Random masks should be known to the sender and helper
            assert_eq!(*result_class.get(&(0, 6)).unwrap(), share_r_sh);
            assert_eq!(*result_class.get(&(0, 7)).unwrap(), share_r_sh);
            // Masked input i0 should be known only to the sender
            assert_eq!(*result_class.get(&(0, 8)).unwrap(), private_class);
            // The sender sends masked i0 to the receiver
            assert_eq!(*result_class.get(&(0, 9)).unwrap(), share_h_rs);
            // Masked input i1 should be known only to the sender
            assert_eq!(*result_class.get(&(0, 10)).unwrap(), private_class);
            // The sender sends masked i1 to the receiver
            assert_eq!(*result_class.get(&(0, 11)).unwrap(), share_h_rs);
            // r1 - r0 is known to the sender and receiver
            assert_eq!(*result_class.get(&(0, 12)).unwrap(), share_r_sh);
            // (r1 - r0) * b is known only to the helper
            assert_eq!(*result_class.get(&(0, 13)).unwrap(), private_class);
            // rb = (r1 - r0) * b + r0 is known only to the helper
            assert_eq!(*result_class.get(&(0, 14)).unwrap(), private_class);
            // rb is sent to the receiver
            assert_eq!(*result_class.get(&(0, 15)).unwrap(), share_s_rh);
            // (i1 + r1) - (i0 + r0) is known to the receiver and sender
            assert_eq!(*result_class.get(&(0, 16)).unwrap(), share_h_rs);
            // ((i1 + r1) - (i0 + r0)) * b is known only to the receiver
            assert_eq!(*result_class.get(&(0, 17)).unwrap(), private_class);
            // ib + rb = ((i1 + r1) - (i0 + r0)) * b + (i0 + r0) is known only to the receiver
            assert_eq!(*result_class.get(&(0, 18)).unwrap(), private_class);
            // ib is known only to the receiver
            assert_eq!(*result_class.get(&(0, 19)).unwrap(), private_class);

            // No more nodes should be
            assert!(result_class.get(&(0, 20)).is_none());

            // Check evaluation
            let result = random_evaluate(
                inlined_c.get_main_graph()?,
                vec![
                    Value::from_flattened_array(&[10, 20], INT32)?,
                    Value::from_flattened_array(&[-10, -20], INT32)?,
                    Value::from_flattened_array(&[0, 1], BIT)?,
                ],
            )?;
            assert_eq!(
                result.to_flattened_array_i32(array_type(vec![2], INT32))?,
                vec![10, -20]
            );
            Ok(())
        };

        roles_helper(1, 0).unwrap();
        roles_helper(0, 1).unwrap();
        roles_helper(1, 2).unwrap();
        roles_helper(2, 1).unwrap();
        roles_helper(0, 2).unwrap();
        roles_helper(2, 0).unwrap();
    }
}
