use crate::custom_ops::CustomOperationBody;
use crate::data_types::{array_type, tuple_type, vector_type, Type, BIT, UINT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph, NodeAnnotation};
use crate::ops::utils::{pull_out_bits, put_in_bits, zeros};

use serde::{Deserialize, Serialize};

use super::mpc_compiler::{KEY_LENGTH, PARTIES};

/// Adds a node returning hash values of an input array of binary strings using provided hash functions.
///
/// Hash functions are defined as an array of binary matrices.
/// The hash of an input string is a product of one of these matrices and this string.
/// Hence, the last dimension of these matrices should coincide with the length of input strings.
///
/// If the input array has shape `[..., n, b]` and hash matrices are given as an `[h, m, b]`-array,
/// then the hash map is an array of shape `[..., h, n]`.
/// The hash table element with index `[..., h, i]` is equal to `j` if the `[..., i]`-th `b`-bit input string is hashed to `j` by the `h`-th hash function.
///
/// When used within a PSI protocol, the hash functions should be the same as those used for Cuckoo hashing.    
///
/// **WARNING**: this function should not be used before MPC compilation.
///
/// # Custom operation arguments
///
/// - input array of binary strings of shape [..., n, b]
/// - random binary [h, m, b]-matrix.
///
/// # Custom operation returns
///
/// hash table of shape [..., h, n] containing UINT64 elements
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct SimpleHash;

#[typetag::serde]
impl CustomOperationBody for SimpleHash {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() != 2 {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            panic!("SimpleHash should have 2 inputs.");
        }

        let input_type = argument_types[0].clone();
        let hash_type = argument_types[1].clone();

        if !matches!(input_type, Type::Array(_, BIT)) {
            return Err(runtime_error!(
                "SimpleHash can't be applied to a non-binary arrays"
            ));
        }
        let input_shape = input_type.get_shape();
        if input_shape.len() < 2 {
            return Err(runtime_error!(
                "Input shape must have at least 2 dimensions"
            ));
        }
        if !matches!(hash_type, Type::Array(_, BIT)) {
            return Err(runtime_error!(
                "SimpleHash needs a binary array as a hash matrix"
            ));
        }
        let hash_shape = hash_type.get_shape();
        if hash_shape.len() != 3 {
            return Err(runtime_error!("Hash array should have 3 dimensions"));
        }
        if hash_shape[1] > 63 {
            return Err(runtime_error!(
                "Hash map is too big. Decrease the number of rows of hash matrices"
            ));
        }
        let input_element_length = input_shape[input_shape.len() - 1];
        if hash_shape[2] != input_element_length {
            return Err(runtime_error!(
                "Hash matrix accepts bitstrings of length {}, but input strings are of length {}",
                hash_shape[2],
                input_element_length
            ));
        }

        let g = context.create_graph()?;

        let input_array = g.input(input_type.clone())?;
        let hash_matrices = g.input(hash_type.clone())?;

        let hash_shape = hash_type.get_shape();

        let mut extended_shape = input_type.get_shape();
        extended_shape.insert(extended_shape.len() - 1, 1);

        // For each subarray and for each hash function, the output hash map contains hashes of input bit strings
        let input_shape = input_type.get_shape();
        let mut single_hash_table_shape = input_shape[0..input_shape.len() - 1].to_vec();
        single_hash_table_shape.push(hash_shape[1]);

        // Multiply hash matrices of shape [h, m, b] by input strings of shape [..., n, b].
        // In Einstein notation, ...nb, hmb -> ...hnm.

        // Change the shape of hash_matrices from [h,m,b] to [b, h*m]
        let hash_matrices_for_matmul = hash_matrices
            .reshape(array_type(
                vec![hash_shape[0] * hash_shape[1], hash_shape[2]],
                BIT,
            ))?
            .permute_axes(vec![1, 0])?;

        // The result shape is [..., n, h*m]
        let mut hash_tables = input_array.matmul(hash_matrices_for_matmul)?;

        // Reshape to [..., n,  h, m]
        let mut split_by_hash_shape = input_shape[0..input_shape.len() - 1].to_vec();
        split_by_hash_shape.extend_from_slice(&hash_shape[0..2]);
        hash_tables = hash_tables.reshape(array_type(split_by_hash_shape.clone(), BIT))?;

        // Transpose to [..., h, n, m]
        let len_output_shape = split_by_hash_shape.len() as u64;
        let mut permuted_axes: Vec<u64> = (0..len_output_shape).collect();
        permuted_axes[len_output_shape as usize - 3] = len_output_shape - 2;
        permuted_axes[len_output_shape as usize - 2] = len_output_shape - 3;
        hash_tables = hash_tables.permute_axes(permuted_axes)?;

        hash_tables = pull_out_bits(hash_tables)?;
        let hash_suffix_type = hash_tables.get_type()?.get_shape()[1..].to_vec();
        let num_zeros = 64 - hash_shape[1];
        let zeros_type = vector_type(num_zeros, array_type(hash_suffix_type.clone(), BIT));
        let zeros = zeros(&g, zeros_type)?;

        hash_tables = g
            .create_tuple(vec![hash_tables.array_to_vector()?, zeros])?
            .reshape(vector_type(64, array_type(hash_suffix_type, BIT)))?
            .vector_to_array()?;

        hash_tables = put_in_bits(hash_tables)?.b2a(UINT64)?;

        hash_tables.set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "SimpleHash".to_owned()
    }
}

/// Adds a node that permutes an array shared between Sender and Programmer using a permutation known to Programmer.
/// The output shares are returned only to Receiver and Programmer.
///
/// Input shares are assumed to be a tuple of 2-out-of-2 shares.
/// Each share must be a named tuple containing arrays of binary strings.
/// So databases converted to such named tuples are handled column-wise.
///
/// The protocol follows the Permute protocol from <https://eprint.iacr.org/2019/518.pdf>.
/// Assume that Sender and Programmer have shares `X_s` and `X_p`, respectively.
/// 1. Programmer creates a random composition of its permutation `perm = perm_r * perm_s`,
/// where `perm_r` and `perm_s` are random permutations sent to Receiver and Sender.
/// 2. Programmer and Sender generate a random mask S of the same type as one input share.
/// 3. Programmer and Receiver generate a random mask T of the same type as one input share.
/// 4. Sender computes `B = perm_s(X_s) - S` and sends it to Receiver
/// 5. Receiver computes its share of the output `Y_r = perm_r(B) - T`.
/// 6. Programmer computes its share of the output `Y_p = perm_r(S) + T + perm(X_p)`.
///
/// **WARNING**: this function should not be used before MPC compilation.
///
/// # Custom operation arguments
///
/// - tuple of 2-out-of-2 shares owned by Sender and Programmer
/// - permutation array known to Programmer
/// - tuple of 3 PRF keys used for multiplication
///
/// # Custom operation returns
///
/// Tuple of permuted 2-out-of-2 shares known to Receiver and Programmer
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct PermutationMPC {
    pub sender_id: u64,
    pub programmer_id: u64, // The receiver ID is defined automatically
}

#[typetag::serde]
impl CustomOperationBody for PermutationMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // Check permutation and input types
        if argument_types.len() != 3 {
            panic!("Permutation protocol should have 3 input types");
        }
        let shares_t = argument_types[0].clone();
        let (num_entries, column_header_types) =
            if let Type::Tuple(shares_type_vector) = shares_t.clone() {
                if shares_type_vector.len() != 2 {
                    panic!("There should be only 2 shares in the input tuple");
                }
                let share_t = (*shares_type_vector[0]).clone();
                if share_t != (*shares_type_vector[1]).clone() {
                    panic!("Input shares must be of the same type");
                }
                if let Type::NamedTuple(column_string_type_vector) = share_t {
                    let mut num_column_entries = 0;
                    for v in column_string_type_vector.iter() {
                        let column_type = (*v.1).clone();
                        let column_shape = column_type.get_shape();
                        if !column_type.is_array() {
                            panic!("Column must be an array");
                        }
                        if num_column_entries == 0 {
                            num_column_entries = column_shape[0];
                        }
                        if num_column_entries != column_shape[0] {
                            panic!("Number of entries should be the same in all columns");
                        }
                    }
                    (num_column_entries, column_string_type_vector)
                } else {
                    panic!("Each share must be a named tuple");
                }
            } else {
                panic!("Input shares must be a tuple of 2 elements");
            };
        let permutation_t = argument_types[1].clone();
        if !permutation_t.is_array() {
            panic!("Permutation must be an array");
        }
        if permutation_t.get_shape()[0] > num_entries {
            panic!("Permutation length must be smaller than the number of entries");
        }
        let prf_t = argument_types[2].clone();
        let expected_key_type = tuple_type(vec![array_type(vec![KEY_LENGTH], BIT); 3]);
        if prf_t != expected_key_type {
            panic!(
                "PRF key type should be a tuple of 3 binary arrays of length {}",
                KEY_LENGTH
            );
        }
        if self.sender_id >= PARTIES as u64 {
            panic!("Sender ID is incorrect");
        }
        if self.programmer_id >= PARTIES as u64 {
            panic!("Programmer ID is incorrect");
        }
        if self.sender_id == self.programmer_id {
            panic!("Programmer ID should be different from the Sender ID")
        }

        let sender_id = self.sender_id;
        let programmer_id = self.programmer_id;
        // This is correct only if PARTIES = 3.
        let receiver_id = PARTIES as u64 - sender_id - programmer_id;

        let g = context.create_graph()?;

        let shares = g.input(shares_t)?;
        let permutation = g.input(permutation_t)?;

        let mut sender_perm = g.random_permutation(num_entries)?;
        let inverse_sender_perm = sender_perm.inverse_permutation()?;
        // Composition permutation(inverse_sender_perm())
        let mut receiver_perm = inverse_sender_perm.gather(permutation.clone(), 0)?;

        // Programmer sends permutations to Sender and Receiver
        sender_perm = sender_perm
            .nop()?
            .add_annotation(NodeAnnotation::Send(programmer_id, sender_id))?;
        receiver_perm = receiver_perm
            .nop()?
            .add_annotation(NodeAnnotation::Send(programmer_id, receiver_id))?;

        // Generate randomness between Sender and Programmer, Programmer and Receiver (PRF keys are needed)
        let prf_keys = g.input(prf_t)?;

        // Choose PRF keys known to Sender and Programmer, Programmer and Receiver.
        // If key is known to parties A and B, then it must be unknown to party C.
        // If C has ID = k, then it knows keys prf_keys[k] and prf_keys[(k+1)%3], but has no clue about prf_keys[(k-1)%3]
        let key_s_p_index = ((receiver_id as usize + PARTIES - 1) % PARTIES) as u64;
        let prf_key_s_p = prf_keys.tuple_get(key_s_p_index)?;
        let key_p_r_index = ((sender_id as usize + PARTIES - 1) % PARTIES) as u64;
        let prf_key_p_r = prf_keys.tuple_get(key_p_r_index)?;

        let sender_share = shares.tuple_get(0)?;
        let programmer_share = shares.tuple_get(1)?;
        let mut receiver_columns = vec![];
        let mut programmer_columns = vec![];
        for column_header_type in column_header_types {
            let column_header = column_header_type.0;
            // Permute the column share of Sender and mask it
            // Select a column
            let sender_share_column = sender_share.named_tuple_get(column_header.clone())?;
            // Permute the column
            let sender_share_column_permuted =
                sender_share_column.gather(sender_perm.clone(), 0)?;
            // Generate a random column mask known to Sender and Programmer
            let sender_column_mask = g.prf(
                prf_key_s_p.clone(),
                0,
                sender_share_column_permuted.get_type()?,
            )?;
            // Mask the column
            let mut sender_share_column_masked =
                sender_share_column_permuted.subtract(sender_column_mask.clone())?;
            // Send the result to Receiver
            sender_share_column_masked = sender_share_column_masked
                .nop()?
                .add_annotation(NodeAnnotation::Send(sender_id, receiver_id))?;
            // Compute the column share of Receiver
            // Permute Sender's masked share
            let mut receiver_result_column =
                sender_share_column_masked.gather(receiver_perm.clone(), 0)?;
            // Generate a random column mask known to Receiver and Programmer
            let receiver_mask =
                g.prf(prf_key_p_r.clone(), 0, receiver_result_column.get_type()?)?;
            // Mask the column
            receiver_result_column = receiver_result_column.subtract(receiver_mask.clone())?;
            // Compute the share of Programmer
            // Select a column
            let programmer_share_column =
                programmer_share.named_tuple_get(column_header.clone())?;
            // Permute Sender's mask (which is known to Programmer) and its input share
            // Then, sum these together with Receiver's mask
            let programmer_result_column = sender_column_mask
                .gather(receiver_perm.clone(), 0)?
                .add(receiver_mask)?
                .add(programmer_share_column.gather(permutation.clone(), 0)?)?;

            receiver_columns.push((column_header.clone(), receiver_result_column));
            programmer_columns.push((column_header, programmer_result_column));
        }
        let receiver_result_share = g.create_named_tuple(receiver_columns)?;
        let programmer_result_share = g.create_named_tuple(programmer_columns)?;

        g.create_tuple(vec![receiver_result_share, programmer_result_share])?
            .set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "Permutation(sender:{},programming:{})",
            self.sender_id, self.programmer_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::scalar_type;
    use crate::data_types::ArrayShape;
    use crate::data_types::INT16;
    use crate::data_types::INT32;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::inline::inline_ops::inline_operations;
    use crate::inline::inline_ops::InlineConfig;
    use crate::inline::inline_ops::InlineMode;
    use crate::mpc::mpc_compiler::IOStatus;
    use crate::mpc::mpc_equivalence_class::generate_equivalence_class;
    use crate::mpc::mpc_equivalence_class::vector_class;
    use crate::mpc::mpc_equivalence_class::EquivalenceClasses;

    fn simple_hash_helper(
        input_shape: ArrayShape,
        hash_shape: ArrayShape,
        inputs: Vec<Value>,
    ) -> Result<Vec<u64>> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(array_type(input_shape.clone(), BIT))?;
        let hash_matrix = g.input(array_type(hash_shape.clone(), BIT))?;
        let o = g.custom_op(CustomOperation::new(SimpleHash), vec![i, hash_matrix])?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?.context;
        let result_value = random_evaluate(mapped_c.get_main_graph()?, inputs)?;
        let mut result_shape = input_shape[0..input_shape.len() - 1].to_vec();
        result_shape.insert(0, hash_shape[0]);
        let result_type = array_type(result_shape, UINT64);
        result_value.to_flattened_array_u64(result_type)
    }

    fn simple_hash_helper_fails(input_t: Type, hash_t: Type) -> Result<()> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(input_t)?;
        let hash_matrix = g.input(hash_t)?;
        let o = g.custom_op(CustomOperation::new(SimpleHash), vec![i, hash_matrix])?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        run_instantiation_pass(c)?;
        Ok(())
    }

    #[test]
    fn test_simple_hash() {
        || -> Result<()> {
            // no collision
            {
                // [2,3]-array
                let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 1], BIT)?;
                // [3,2,3]-array
                let hash_matrix = Value::from_flattened_array(
                    &[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    BIT,
                )?;
                // output [3,2]-array
                let expected = vec![0, 1, 0, 2, 3, 2];
                assert_eq!(
                    simple_hash_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix])?,
                    expected
                );
            }
            // collisions
            {
                // [2,3]-array
                let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 0], BIT)?;
                // [3,2,3]-array
                let hash_matrix = Value::from_flattened_array(
                    &[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    BIT,
                )?;
                // output [3,2]-array
                let expected = vec![0, 0, 0, 0, 3, 0];
                assert_eq!(
                    simple_hash_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix])?,
                    expected
                );
            }
            {
                // [2,2,2]-array
                let input = Value::from_flattened_array(&[1, 0, 0, 0, 1, 1, 0, 1], BIT)?;
                // [2,3,2]-array
                let hash_matrix =
                    Value::from_flattened_array(&[1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], BIT)?;
                // output [2,2,2]-array
                let expected = vec![3, 0, 2, 0, 7, 4, 7, 5];
                assert_eq!(
                    simple_hash_helper(vec![2, 2, 2], vec![2, 3, 2], vec![input, hash_matrix])?,
                    expected
                );
            }
            // malformed input
            {
                let input_t = scalar_type(BIT);
                let hash_t = array_type(vec![2, 3, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], UINT64);
                let hash_t = array_type(vec![2, 3, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![4], BIT);
                let hash_t = array_type(vec![2, 3, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = scalar_type(BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = array_type(vec![3, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = array_type(vec![2, 3, 4], UINT64);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = array_type(vec![2, 64, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = array_type(vec![2, 3, 5], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_permutation() {
        let data_helper = |a_type: Type,
                           b_type: Type,
                           a_values: &[u64],
                           b_values: &[u64],
                           permutation_values: &[u64],
                           a_expected: &[u64],
                           b_expected: &[u64]|
         -> Result<()> {
            // test correct inputs
            let roles_helper = |sender_id: u64, programmer_id: u64| -> Result<()> {
                let c = create_context()?;

                let g = c.create_graph()?;

                let column_a = g.input(a_type.clone())?;
                let column_b = g.input(b_type.clone())?;

                // Generate PRF keys
                let key_t = array_type(vec![KEY_LENGTH], BIT);
                // PRF key known to parties 2 and 0
                let key_2_0 = g
                    .random(key_t.clone())?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(2, 0))?;
                // PRF key known to parties 0 and 1
                let key_0_1 = g
                    .random(key_t.clone())?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(0, 1))?;
                // PRF key known to parties 1 and 2
                let key_1_2 = g
                    .random(key_t.clone())?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(1, 2))?;
                let keys = g.create_tuple(vec![key_2_0, key_0_1, key_1_2])?;
                // PRF key known only to Sender.
                let key_s = g.random(key_t.clone())?;
                // Split input into two shares between Sender and Programmer
                // Sender generates Programmer's shares
                let column_a_programmer_share = g.prf(key_s.clone(), 0, a_type.clone())?;
                let column_b_programmer_share = g.prf(key_s.clone(), 0, b_type.clone())?;
                // Sender computes its shares
                let column_a_sender_share = column_a.subtract(column_a_programmer_share.clone())?;
                let column_b_sender_share = column_b.subtract(column_b_programmer_share.clone())?;

                // Sender packs shares in named tuples and send one of them to Programmer
                let programmer_share = g
                    .create_named_tuple(vec![
                        ("a".to_owned(), column_a_programmer_share),
                        ("b".to_owned(), column_b_programmer_share),
                    ])?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(sender_id, programmer_id))?;
                let sender_share = g.create_named_tuple(vec![
                    ("a".to_owned(), column_a_sender_share),
                    ("b".to_owned(), column_b_sender_share),
                ])?;

                // Pack shares together
                let shares = g.create_tuple(vec![sender_share, programmer_share])?;

                // Permutation input
                let permutation =
                    g.input(array_type(vec![permutation_values.len() as u64], UINT64))?;

                // Permuted shares
                let permuted_shares = g.custom_op(
                    CustomOperation::new(PermutationMPC {
                        sender_id,
                        programmer_id,
                    }),
                    vec![shares, permutation, keys],
                )?;

                // Sum permuted shares
                let receiver_permuted_share = permuted_shares.tuple_get(0)?;
                let programmer_permuted_share = permuted_shares.tuple_get(1)?;

                let permuted_column_a = receiver_permuted_share
                    .named_tuple_get("a".to_owned())?
                    .add(programmer_permuted_share.named_tuple_get("a".to_owned())?)?;
                let permuted_column_b = receiver_permuted_share
                    .named_tuple_get("b".to_owned())?
                    .add(programmer_permuted_share.named_tuple_get("b".to_owned())?)?;

                // Combine permuted columns
                g.create_tuple(vec![permuted_column_a, permuted_column_b])?
                    .set_as_output()?;

                g.finalize()?;
                g.set_as_main()?;
                c.finalize()?;

                let instantiated_c = run_instantiation_pass(c)?.context;
                let inlined_c = inline_operations(
                    instantiated_c,
                    InlineConfig {
                        default_mode: InlineMode::Simple,
                        ..Default::default()
                    },
                )?;

                let result_hashmap = generate_equivalence_class(
                    inlined_c.clone(),
                    vec![vec![
                        IOStatus::Party(sender_id),
                        IOStatus::Party(sender_id),
                        IOStatus::Party(programmer_id),
                    ]],
                )?;

                let receiver_id = PARTIES as u64 - sender_id - programmer_id;
                let private_class = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);
                // data shared by Sender and Programmer
                let share_r_sp = EquivalenceClasses::Atomic(vec![
                    vec![receiver_id],
                    vec![sender_id, programmer_id],
                ]);
                // data shared by the Receiver and Programmer
                let share_s_rp = EquivalenceClasses::Atomic(vec![
                    vec![sender_id],
                    vec![receiver_id, programmer_id],
                ]);
                // data shared by Receiver and Sender
                let share_p_rs = EquivalenceClasses::Atomic(vec![
                    vec![programmer_id],
                    vec![receiver_id, sender_id],
                ]);
                // data shared by parties 0 and 1
                let share_2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);
                // data shared by parties 1 and 2
                let share_0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
                // data shared by parties 2 and 0
                let share_1_20 = EquivalenceClasses::Atomic(vec![vec![1], vec![2, 0]]);

                let private_pair = vector_class(vec![private_class.clone(); 2]);
                let programmers_share_class = vector_class(vec![share_r_sp.clone(); 2]);

                let expected_classes = vec![
                    // both inputs should be known only to Sender
                    private_class.clone(),
                    private_class.clone(),
                    // First PRF key
                    private_class.clone(),
                    share_1_20.clone(),
                    // Second PRF key
                    private_class.clone(),
                    share_2_01.clone(),
                    // Third PRF key
                    private_class.clone(),
                    share_0_12.clone(),
                    // All PRF keys
                    vector_class(vec![
                        share_1_20.clone(),
                        share_2_01.clone(),
                        share_0_12.clone(),
                    ]),
                    // PRF key known to Sender
                    private_class.clone(),
                    // Programmer's input shares
                    private_class.clone(),
                    private_class.clone(),
                    // Sender's input shares
                    private_class.clone(),
                    private_class.clone(),
                    // Programmer's share
                    private_pair.clone(),
                    programmers_share_class.clone(),
                    // Sender's share
                    private_pair.clone(),
                    // Tuple of both shares
                    vector_class(vec![private_pair.clone(), programmers_share_class.clone()]),
                    // Permutation input
                    private_class.clone(),
                    // Sender's permutation generated by Programmer
                    private_class.clone(),
                    // Inverse of Sender's permutation
                    private_class.clone(),
                    // Receiver's permutation generated by Programmer
                    private_class.clone(),
                    // Sender's permutation after sending to Sender
                    share_r_sp.clone(),
                    // Receiver's permutation after sending to Receiver
                    share_s_rp.clone(),
                    // PRF key known to Sender and Programmer
                    share_r_sp.clone(),
                    // PRF key known to Receiver and Programmer
                    share_s_rp.clone(),
                    // Sender's share
                    private_pair.clone(),
                    // Programmer's share
                    programmers_share_class.clone(),
                    // Sender's share of the first column
                    private_class.clone(),
                    // Permuted Sender's share of the first column
                    private_class.clone(),
                    // Random mask known to Sender and Programmer
                    share_r_sp.clone(),
                    // Masked permuted Sender's share of the first column
                    private_class.clone(),
                    // Masked permuted Sender's share of the first column sent to Receiver
                    share_p_rs.clone(),
                    // Receiver permutes the above share
                    private_class.clone(),
                    // Random mask known to Receiver and Programmer
                    share_s_rp.clone(),
                    // Receiver's resulting share of the permuted first column
                    private_class.clone(),
                    // Sender's share of the first column (since Sender shared data)
                    share_r_sp.clone(),
                    // Permutation of Sender's mask
                    private_class.clone(),
                    // Sum of the permuted Sender's mask and Receiver's mask
                    private_class.clone(),
                    // Permutation of Programmer's share of the first column
                    private_class.clone(),
                    // Programmer's resulting share of the permuted first column
                    private_class.clone(),
                    // Sender's share of the second column
                    private_class.clone(),
                    // Permuted Sender's share of the second column
                    private_class.clone(),
                    // Random mask known to Sender and Programmer
                    share_r_sp.clone(),
                    // Masked permuted Sender's share of the second column
                    private_class.clone(),
                    // Masked permuted Sender's share of the second column sent to Receiver
                    share_p_rs.clone(),
                    // Receiver permutes the above share
                    private_class.clone(),
                    // Random mask known to Receiver and Programmer
                    share_s_rp.clone(),
                    // Receiver's resulting share of the permuted second column
                    private_class.clone(),
                    // Sender's share of the second column (since Sender shared data)
                    share_r_sp,
                    // Permutation of Sender's mask
                    private_class.clone(),
                    // Sum of the permuted Sender's mask and Receiver's mask
                    private_class.clone(),
                    // Permutation of Programmer's share of the second column
                    private_class.clone(),
                    // Programmer's resulting share of the permuted second column
                    private_class.clone(),
                    // Receiver's result share of the named tuple
                    private_pair.clone(),
                    // Programmer's result share of the named tuple
                    private_pair.clone(),
                    // Both shares combined (the output of the protocol)
                    vector_class(vec![private_pair.clone(); 2]),
                ];
                let mut result_classes = vec![];
                for i in 0..expected_classes.len() as u64 {
                    result_classes.push((*result_hashmap.get(&(0, i)).unwrap()).clone());
                }
                assert_eq!(result_classes, expected_classes);

                // Check evaluation
                let result = random_evaluate(
                    inlined_c.get_main_graph()?,
                    vec![
                        Value::from_flattened_array(a_values.clone(), a_type.get_scalar_type())?,
                        Value::from_flattened_array(b_values.clone(), b_type.get_scalar_type())?,
                        Value::from_flattened_array(permutation_values.clone(), UINT64)?,
                    ],
                )?;
                let mut result_a_shape = a_type.get_shape();
                result_a_shape[0] = permutation_values.len() as u64;
                let result_a_type = array_type(result_a_shape, a_type.get_scalar_type());

                let mut result_b_shape = b_type.get_shape();
                result_b_shape[0] = permutation_values.len() as u64;
                let result_b_type = array_type(result_b_shape, b_type.get_scalar_type());

                let result_a =
                    result.to_vector()?[0].to_flattened_array_u64(result_a_type.clone())?;
                let result_b =
                    result.to_vector()?[1].to_flattened_array_u64(result_b_type.clone())?;
                assert_eq!(&result_a, a_expected.clone());
                assert_eq!(&result_b, b_expected.clone());
                Ok(())
            };
            roles_helper(1, 0)?;
            roles_helper(0, 1)?;
            roles_helper(1, 2)?;
            roles_helper(2, 1)?;
            roles_helper(0, 2)?;
            roles_helper(2, 0)?;
            Ok(())
        };

        data_helper(
            array_type(vec![5], INT32),
            array_type(vec![5], INT16),
            &[1, 2, 3, 4, 5],
            &[10, 20, 30, 40, 50],
            &[1, 0, 3, 4, 2],
            &[2, 1, 4, 5, 3],
            &[20, 10, 40, 50, 30],
        )
        .unwrap();

        data_helper(
            array_type(vec![5], INT32),
            array_type(vec![5], UINT64),
            &[1, 2, 3, 4, 5],
            &[10, 20, 30, 40, 50],
            &[0, 1, 2],
            &[1, 2, 3],
            &[10, 20, 30],
        )
        .unwrap();

        data_helper(
            array_type(vec![5, 2], BIT),
            array_type(vec![5], UINT64),
            &[0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
            &[10, 20, 30, 40, 50],
            &[0, 2, 4, 1],
            &[0, 0, 1, 0, 0, 1, 0, 1],
            &[10, 30, 50, 20],
        )
        .unwrap();
    }
}
