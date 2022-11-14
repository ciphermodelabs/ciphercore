use crate::custom_ops::CustomOperationBody;
use crate::data_types::{array_type, vector_type, Type, BIT};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, SliceElement};
use crate::ops::utils::{pull_out_bits, put_in_bits, zeros};

use serde::{Deserialize, Serialize};

pub(super) const LOW_MC_KEY_SIZE: u64 = 128;

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum LowMCBlockSize {
    SIZE80,
    SIZE128,
}

/// Implements LowMC block cipher encryption according to [the original LowMC publication (Section 3)](https://eprint.iacr.org/2016/687.pdf).
///
/// At least two inputs should be provided:
/// - input,
/// - encryption key.
///
/// The first encryption step is to add the first pre-computed round key to the input.
/// Then, the series of the rounds is performed with the following subroutines:
/// - substitution,
/// - multiplication by a pre-computed random matrix,
/// - addition of a pre-computed random bitstring,
/// - addition of a pre-computed round key.
///
/// The LowMC parameters are fixed to 128-bit block and key sizes.
/// The number of substituted bits and encryption rounds should support 128-bits of security.
/// Using [this script](https://github.com/LowMC/lowmc/blob/master/determine_rounds.py), we identified the following parameters:
///
/// |s_boxes_per_round | rounds |
/// |:----------------:|:------:|
/// | 10               |20 (Picnic parameters)     |
/// | 11               |19      |
/// | 12               |18      |
/// | 13               |17      |
/// | 14               |16      |
/// | 15               |14      |
/// | 16               |14      |
/// | 18               |13      |
/// | 19               |13      |
/// | 20               |13      |
/// | 21               |13      |
/// | 22               |12      |
/// | 25               |11      |
/// ...
/// | 42               |11      |
/// See [the Picnic specification (Section 4)](https://github.com/microsoft/Picnic/blob/master/spec/spec-v3.0.pdf) for more details.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct LowMC {
    // Number of bit triples affected by a single substitution round
    // It should not exceed LOW_MC_BLOCK_SIZE/3.
    pub s_boxes_per_round: u64,
    // Number of encryption rounds (max 20).
    pub rounds: u64,
    pub block_size: LowMCBlockSize,
}

#[typetag::serde]
impl CustomOperationBody for LowMC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // Size of a single LowMC block (typically)
        let block_size = match self.block_size {
            LowMCBlockSize::SIZE128 => 128,
            LowMCBlockSize::SIZE80 => 80,
        };
        // Length of an encryption key
        let key_size = LOW_MC_KEY_SIZE;

        // Check that the number of triples affected by a single substitution round doesn't exceed the number of bits in the block
        if self.s_boxes_per_round > block_size / 3 {
            return Err(runtime_error!(
                "The number of S-boxes must be between 10 and 42"
            ));
        }
        // Check that the number of encryption rounds doesn't exceed 20.
        // The pre-generated random matrices support at most 20 rounds.
        if self.rounds > 20 {
            return Err(runtime_error!("The number of rounds can't exceed 20"));
        }

        if argument_types.len() != 2 {
            return Err(runtime_error!(
                "LowMC should have 2 inputs: input and an encryption key"
            ));
        }

        if argument_types[0].get_scalar_type() != BIT {
            return Err(runtime_error!("Input of LowMC must be binary"));
        }

        let input_shape = argument_types[0].get_shape();
        let input_element_len = input_shape[input_shape.len() - 1];
        if input_element_len > block_size {
            return Err(runtime_error!(
                "Input bitstrings should be of length {}",
                block_size
            ));
        }

        if argument_types[1] != array_type(vec![key_size], BIT) {
            return Err(runtime_error!(
                "LowMC key must be a binary array of length {}",
                key_size
            ));
        }

        let g = context.create_graph()?;

        let input = g.input(argument_types[0].clone())?;
        let key = g.input(argument_types[1].clone())?;

        // Pad input with zeros
        let padded_input = if input_element_len < block_size {
            let length_to_pad = block_size - input_element_len;
            let bits = pull_out_bits(input)?.array_to_vector()?;
            let zeros_shape = input_shape[0..input_shape.len() - 1].to_vec();
            let zeros_type = vector_type(length_to_pad, array_type(zeros_shape.clone(), BIT));
            put_in_bits(
                g.create_tuple(vec![bits, zeros(&g, zeros_type)?])?
                    .reshape(vector_type(block_size, array_type(zeros_shape, BIT)))?
                    .vector_to_array()?,
            )?
        } else {
            input
        };

        // First, we initialize the internal constants of LowMC for each round, namely:
        // Once initialized, these constants must be fixed for all LowMC nodes in the graph.
        // These constants are reused from from [the Picnic implementation](https://github.com/microsoft/Picnic/blob/master/lowmc_constants.c) (commit c56abd3baf6629c1f1c122929ebbaac8a9baec15)
        // In the protocol, these constants are supposed to be public.

        // Constants are generated with parameters satisfying an 128-bit security level, namely:
        // - the block and key sizes are both 128
        // - the number of rounds is 20

        let (linear_matrices_value, round_constants_value, key_matrices_value) = match self
            .block_size
        {
            LowMCBlockSize::SIZE128 => {
                // - a (block_size x block_size) invertible binary matrix for the linear layer,
                let linear_matrices_bytes =
                    include_bytes!("low_mc_constants/linear_layer_matrices128.dat");
                let l_value = Value::from_bytes(
                    linear_matrices_bytes[0..(self.rounds * block_size * block_size / 8) as usize]
                        .to_vec(),
                );

                // - a binary vector of length block_size for the constant addition layer,
                let round_constants_bytes =
                    include_bytes!("low_mc_constants/round_constants128.dat");
                let r_value = Value::from_bytes(
                    round_constants_bytes[0..(self.rounds * block_size / 8) as usize].to_vec(),
                );

                // - a (block_size x key_size) binary matrix of rank min(key_size, block_size) for the key addition layer plus one more matrix of this size to whiten the key.
                let key_matrices_bytes = include_bytes!("low_mc_constants/key_matrices128.dat");
                let k_value = Value::from_bytes(
                    key_matrices_bytes[0..((self.rounds + 1) * block_size * key_size / 8) as usize]
                        .to_vec(),
                );

                (l_value, r_value, k_value)
            }
            LowMCBlockSize::SIZE80 => {
                // - a (block_size x block_size) invertible binary matrix for the linear layer,
                let linear_matrices_bytes =
                    include_bytes!("low_mc_constants/linear_layer_matrices80.dat");
                let l_value = Value::from_bytes(
                    linear_matrices_bytes[0..(self.rounds * block_size * block_size / 8) as usize]
                        .to_vec(),
                );

                // - a binary vector of length block_size for the constant addition layer,
                let round_constants_bytes =
                    include_bytes!("low_mc_constants/round_constants80.dat");
                let r_value = Value::from_bytes(
                    round_constants_bytes[0..(self.rounds * block_size / 8) as usize].to_vec(),
                );

                // - a (block_size x key_size) binary matrix of rank min(key_size, block_size) for the key addition layer plus one more matrix of this size to whiten the key.
                let key_matrices_bytes = include_bytes!("low_mc_constants/key_matrices80.dat");
                let k_value = Value::from_bytes(
                    key_matrices_bytes[0..((self.rounds + 1) * block_size * key_size / 8) as usize]
                        .to_vec(),
                );

                (l_value, r_value, k_value)
            }
        };
        let linear_matrices_type = array_type(vec![self.rounds, block_size, block_size], BIT);
        let round_constants_type = array_type(vec![self.rounds, block_size], BIT);
        let key_matrices_type = array_type(vec![self.rounds + 1, block_size, key_size], BIT);

        let linear_matrices = g.constant(linear_matrices_type, linear_matrices_value)?;
        let round_constants = g.constant(round_constants_type, round_constants_value)?;
        let key_matrices = g.constant(key_matrices_type, key_matrices_value)?;

        // Round keys generated from the master key
        let key_schedule = key_matrices
            .gemm(
                key.reshape(array_type(vec![1, key_size], BIT))?,
                false,
                true,
            )?
            .reshape(array_type(vec![self.rounds + 1, block_size], BIT))?;

        // XOR hashed input with the whitened key (1st element of the key schedule)
        let mut state = padded_input.add(key_schedule.get(vec![0])?)?;
        let state_type = state.get_type()?;
        let state_shape = state_type.get_shape();
        let state_element_shape = state_shape[0..state_shape.len() - 1].to_vec();
        let state_element_type =
            array_type(state_element_shape, state.get_type()?.get_scalar_type());

        // Compute all the rounds
        for round in 0..self.rounds {
            state = pull_out_bits(state)?;

            // Substitution layer
            // Take s_boxes_per_round triples of bits and map each triple (a,b,c) to (a+bc, a+b+ac, a+b+c+ab)
            let a_bits = state.get_slice(vec![SliceElement::SubArray(
                Some(2),
                Some(3 * self.s_boxes_per_round as i64),
                Some(3),
            )])?;
            let b_bits = state.get_slice(vec![SliceElement::SubArray(
                Some(1),
                Some(3 * self.s_boxes_per_round as i64),
                Some(3),
            )])?;
            let c_bits = state.get_slice(vec![SliceElement::SubArray(
                Some(0),
                Some(3 * self.s_boxes_per_round as i64),
                Some(3),
            )])?;

            let new_a_bits = a_bits.add(b_bits.multiply(c_bits.clone())?)?;
            let new_b_bits = a_bits
                .add(b_bits.clone())?
                .add(a_bits.multiply(c_bits.clone())?)?;
            let new_c_bits = a_bits
                .add(b_bits.clone())?
                .add(c_bits)?
                .add(a_bits.multiply(b_bits)?)?;

            // Take the rest of state elements
            let fixed_bits = state.get_slice(vec![SliceElement::SubArray(
                Some(3 * self.s_boxes_per_round as i64),
                None,
                None,
            )])?;

            // Merge substituted bits and fixed bits
            state = g
                .create_tuple(vec![
                    g.zip(vec![
                        new_c_bits.array_to_vector()?,
                        new_b_bits.array_to_vector()?,
                        new_a_bits.array_to_vector()?,
                    ])?,
                    fixed_bits.array_to_vector()?,
                ])?
                .reshape(vector_type(block_size, state_element_type.clone()))?
                .vector_to_array()?;
            state = put_in_bits(state)?;

            // Linear layer: multiply by pre-generated random matrices
            state = state.gemm(linear_matrices.get(vec![round])?, false, true)?;

            // Add the round constant
            state = state.add(round_constants.get(vec![round])?)?;

            // Add the round key
            state = state.add(key_schedule.get(vec![round + 1])?)?;
        }

        state.set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("LowMC({}-{})", self.s_boxes_per_round, self.rounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::random::entropy_test;

    fn helper_with_reference(input: Vec<u8>, expected: Vec<u8>) -> Result<()> {
        let key_size = 128;
        let input_size = 128;

        let input_shape = vec![2, 2, input_size];

        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(array_type(input_shape, BIT))?;
        let key = g.input(array_type(vec![key_size], BIT))?;
        let o = g.custom_op(
            CustomOperation::new(LowMC {
                s_boxes_per_round: 10,
                rounds: 20,
                block_size: LowMCBlockSize::SIZE128,
            }),
            vec![i, key],
        )?;
        o.set_as_output()?;
        g.finalize()?;
        g.set_as_main()?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?;

        let key_value = Value::from_bytes(
            (*b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10").to_vec(),
        );
        let input_value = Value::from_bytes(input);
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![input_value, key_value],
        )?;
        result.access_bytes(|bytes| {
            assert_eq!(bytes, &expected);
            Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn test_low_mc_with_reference() {
        || -> Result<()> {
            // Expected output is generated using the LowMC reference implementation from https://github.com/LowMC/lowmc
            let input = vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            ];
            let expected = vec![
                196, 26, 77, 159, 144, 79, 239, 201, 114, 177, 170, 16, 242, 232, 87, 226, 54, 17,
                2, 143, 191, 198, 219, 85, 136, 213, 61, 45, 85, 161, 47, 226, 41, 50, 219, 76, 17,
                167, 157, 108, 22, 185, 248, 245, 246, 172, 115, 5, 172, 28, 169, 195, 204, 32, 59,
                246, 170, 141, 10, 23, 87, 8, 161, 247,
            ];
            helper_with_reference(input, expected)?;
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_low_mc_randomness() {
        || -> Result<()> {
            let key_size = 128;
            let input_size = 76;

            let input_shape = vec![16, 16, input_size];
            let input_bytes_len = input_shape.iter().product::<u64>() >> 3;

            let c = create_context()?;
            let g = c.create_graph()?;
            let i = g.input(array_type(input_shape.clone(), BIT))?;
            let key = g.input(array_type(vec![key_size], BIT))?;
            let o = g.custom_op(
                CustomOperation::new(LowMC {
                    s_boxes_per_round: 26,
                    rounds: 4,
                    block_size: LowMCBlockSize::SIZE80,
                }),
                vec![i, key],
            )?;
            o.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;
            let mapped_c = run_instantiation_pass(c)?;

            let key_value = Value::from_bytes(
                (*b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10").to_vec(),
            );
            let mut input_bytes = vec![0u8; input_bytes_len as usize];
            for i in 0..input_shape[0].clone() {
                for j in 0..input_shape[1].clone() {
                    for k in 0..input_size / 8 {
                        input_bytes
                            [((i * input_shape[1].clone() + j) * input_size / 8 + k) as usize] =
                            (i * input_shape[1].clone() + j) as u8;
                    }
                }
            }
            let input_value = Value::from_bytes(input_bytes);
            let result = random_evaluate(
                mapped_c.get_context().get_main_graph()?,
                vec![input_value, key_value],
            )?;
            result.access_bytes(|bytes| {
                //Check that bytes are random
                let mut counters = [0; 256];
                for i in 0..bytes.len() {
                    counters[bytes[i] as usize] += 1;
                }
                assert!(entropy_test(counters, bytes.len() as u64));
                Ok(())
            })?;

            Ok(())
        }()
        .unwrap();
    }
}
