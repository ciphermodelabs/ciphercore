use crate::bytes::vec_from_bytes;
use crate::data_types::{get_size_in_bits, get_types_vector, Type, UINT64};
use crate::data_values::Value;
use crate::errors::Result;

use openssl::symm::{Cipher, Crypter, Mode};
use rand::rngs::OsRng;
use rand::RngCore;

/// It is possible that when used during early boot
/// the first call to OsRng will block until the system’s RNG is initialised.
/// It is also possible (though highly unlikely) for OsRng to fail on some platforms,
/// most likely due to system mis-configuration.
pub fn get_bytes_from_os(bytes: &mut [u8]) -> Result<()> {
    OsRng
        .try_fill_bytes(bytes)
        .map_err(|_| runtime_error!("OS random generator failed"))?;
    Ok(())
}

/// Byte size of PRNG seed.
pub const SEED_SIZE: usize = 16;

/// Buffer size for random bytes. Benchmarks show that lowering it decreases performance, while making it higher - doesn't affect it.
const BUFFER_SIZE: usize = 512;

/// Estimation for the start size of the buffer for generic types (it grows to `BUFFER_SIZE` over time).
const INITIAL_BUFFER_SIZE: usize = 64;

/// Cryptographic pseudo-random generator based on AES-128 in the counter mode.
/// If the seed is private, the security is based on the key-recovery hardness assumption of AES
/// and [the PRP/PRF(Prf) switching lemma](https://eprint.iacr.org/2004/331.pdf).
/// Less than 2^64 128-bit strings should be sampled to avoid distinguishing attacks due to birthday paradox.
/// The empirical randomness properties of AES output is shown in "Empirical Evidence Concerning AES"
/// by Peter Hellekalek and Stefan Wegenkittl.
pub struct PRNG {
    aes: Crypter,
    random_source: PrfSession,
}
/// The following implementation is not thread-safe as several copies of PRNG
/// can concurrently access the system random generator
impl PRNG {
    pub fn new(seed: Option<[u8; SEED_SIZE]>) -> Result<PRNG> {
        let err = |_| runtime_error!("Crypter didn't initialize");
        let bytes = match seed {
            Some(bytes) => bytes,
            None => {
                let mut bytes = [0u8; SEED_SIZE];
                get_bytes_from_os(&mut bytes)?;
                bytes
            }
        };
        let mut c =
            Crypter::new(Cipher::aes_128_ecb(), Mode::Encrypt, &bytes, None).map_err(err)?;
        c.pad(false);
        Ok(PRNG {
            aes: c,
            random_source: PrfSession::new(0, BUFFER_SIZE)?,
        })
    }

    pub fn get_random_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        self.random_source
            .generate_random_bytes(&mut self.aes, n as u64)
    }

    fn get_random_key(&mut self) -> Result<[u8; SEED_SIZE]> {
        let bytes = self.get_random_bytes(SEED_SIZE)?;
        let mut new_seed = [0u8; SEED_SIZE];
        new_seed.copy_from_slice(&bytes[0..SEED_SIZE]);
        Ok(new_seed)
    }

    pub fn get_random_value(&mut self, t: Type) -> Result<Value> {
        match t {
            Type::Scalar(_) | Type::Array(_, _) => {
                let bit_size = get_size_in_bits(t)?;
                let byte_size = (bit_size + 7) / 8;
                // the last random byte should contain bits_to_flush zeros
                let bits_to_flush = 8 * byte_size - bit_size;
                let mut bytes = self.get_random_bytes(byte_size as usize)?;
                // Remove unused random bits
                if !bytes.is_empty() {
                    *bytes.last_mut().unwrap() >>= bits_to_flush;
                }
                Ok(Value::from_bytes(bytes))
            }
            Type::Tuple(_) | Type::Vector(_, _) | Type::NamedTuple(_) => {
                let ts = get_types_vector(t)?;
                let mut v = vec![];
                for sub_t in ts {
                    v.push(self.get_random_value((*sub_t).clone())?)
                }
                Ok(Value::from_vector(v))
            }
        }
    }

    // Generates a random 64-bit integer modulo a given modulus.
    // To avoid modulo bias, rejection sampling is performed.
    // Rejection sampling bound is 2^64 - (2^64 mod modulus).
    // Each sampling attempt can succeed with probability 1 - (2^64 mod modulus)/2^64.
    // Thus, the expected number of sampling rounds is  2^64/(2^64 - (2^64 mod modulus)) < 2 according to the geometric distribution.
    //
    // **WARNING**: this function might leak modulus bits. Don't use it if you want to hide the modulus.
    pub fn get_random_in_range(&mut self, modulus: Option<u64>) -> Result<u64> {
        if let Some(m) = modulus {
            let rem = ((u64::MAX % m) + 1) % m;
            let rejection_bound = u64::MAX - rem;
            let mut r;
            loop {
                r = vec_from_bytes(&self.get_random_bytes(8)?, UINT64)?[0];
                if r <= rejection_bound {
                    break;
                }
            }
            Ok(r % m)
        } else {
            Ok(vec_from_bytes(&self.get_random_bytes(8)?, UINT64)?[0])
        }
    }
}

/// Pseudo-random function (Prf/PRF) based on AES-128.
/// PRF keys are sampled via the above PRNG.
/// As for the above PRNG, the security is based on the key-recovery hardness assumption of AES
/// and [the PRP/PRF(Prf) switching lemma](https://eprint.iacr.org/2004/331.pdf).
/// Less than 2^64 128-bit strings should be sampled to avoid distinguishing attacks due to birthday paradox.
/// PRF(Prf) output is extended by computing AES_k(0|input)|...|AES_k(n-1|input)
/// (see e.g. p.16 of [Kolesnikov et al.](https://eprint.iacr.org/2016/799.pdf)).
pub(super) struct Prf {
    aes: Crypter,
}

impl Prf {
    pub fn new(key: Option<[u8; SEED_SIZE]>) -> Result<Prf> {
        let err = |_| runtime_error!("Crypter didn't initialize");
        let key_bytes = match key {
            Some(bytes) => bytes,
            None => {
                let mut gen = PRNG::new(None)?;
                gen.get_random_key()?
            }
        };
        let mut c =
            Crypter::new(Cipher::aes_128_ecb(), Mode::Encrypt, &key_bytes, None).map_err(err)?;
        c.pad(false);
        Ok(Prf { aes: c })
    }

    #[cfg(test)]
    fn output_bytes(&mut self, input: u64, n: u64) -> Result<Vec<u8>> {
        let initial_buffer_size = usize::min(BUFFER_SIZE, n as usize);
        PrfSession::new(input, initial_buffer_size)?.generate_random_bytes(&mut self.aes, n)
    }

    pub(super) fn output_value(&mut self, input: u64, t: Type) -> Result<Value> {
        PrfSession::new(input, INITIAL_BUFFER_SIZE)?.recursively_generate_value(&mut self.aes, t)
    }

    pub(super) fn output_permutation(&mut self, input: u64, n: u64) -> Result<Value> {
        if n > 2u64.pow(30) {
            return Err(runtime_error!("n should be less than 2^30"));
        }
        // For small n, probability of failure is very low, so we can use `n` as the buffer size.
        // For larger n, we start from `BUFFER_SIZE` right away.
        let initial_buffer_size = usize::min(BUFFER_SIZE, n as usize);
        let mut session = PrfSession::new(input, initial_buffer_size)?;
        let mut a: Vec<u64> = (0..n).collect();
        for i in 1..n {
            let j = session.generate_u32_in_range(&mut self.aes, i as u32 + 1)?;
            a.swap(i as usize, j as usize);
        }
        Value::from_flattened_array_u64(&a, UINT64)
    }
}

// Helper struct for `Prf` that produces random values of various types from random bytes that
// are generated using the `aes` crypter, and the initial `input` value that is incremented
// as more bytes are needed.
struct PrfSession {
    input: u128,
    buffer: Vec<u8>,
    next_byte: usize,
    // Buffer size needs to be a multiple of the cipher block size: 16.
    current_buffer_size: usize,
    next_buffer_size: usize,
}

impl PrfSession {
    pub fn new(input: u64, initial_buffer_size: usize) -> Result<Self> {
        // Round up to the nearest multiple of 16.
        let initial_buffer_size = (initial_buffer_size + 15) / 16 * 16;
        Ok(Self {
            input: (input as u128) << 64,
            // Note: we'll drop the "leftover" bytes when PrfSession is destroyed. They are not useful, but if there
            // are many small PRFs, this can be wasteful.
            buffer: vec![0u8; initial_buffer_size + Cipher::aes_128_cbc().block_size()],
            next_byte: initial_buffer_size,
            current_buffer_size: initial_buffer_size,
            next_buffer_size: initial_buffer_size,
        })
    }

    fn generate_one_batch(&mut self, aes: &mut Crypter) -> Result<()> {
        let mut i_bytes = vec![0u8; self.next_buffer_size];
        for i in (0..i_bytes.len()).step_by(16) {
            i_bytes[i..i + 16].copy_from_slice(&self.input.to_le_bytes());
            self.input = self.input.wrapping_add(1);
        }
        let buffer_len = self.next_buffer_size + Cipher::aes_128_cbc().block_size();
        if buffer_len != self.buffer.len() {
            self.buffer.resize(buffer_len, 0);
        }
        let count = aes
            .update(&i_bytes, &mut self.buffer)
            .map_err(|_| runtime_error!("Crypter didn't manage to update"))?;
        // finalization of Crypter is unnecessary since padding is turned off
        // check here https://www.openssl.org/docs/manmaster/man3/EVP_CipherUpdate.html
        if count != self.next_buffer_size {
            return Err(runtime_error!(
                "AES encryption returned a wrong number of bytes"
            ));
        }
        self.current_buffer_size = self.next_buffer_size;
        if self.next_buffer_size < BUFFER_SIZE {
            self.next_buffer_size = usize::min(BUFFER_SIZE, self.next_buffer_size * 2);
        }
        self.next_byte = 0;
        Ok(())
    }

    fn generate_random_bytes(&mut self, aes: &mut Crypter, n: u64) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; n as usize];
        self.fill_random_bytes(aes, bytes.as_mut_slice())?;
        Ok(bytes)
    }

    fn fill_random_bytes(&mut self, aes: &mut Crypter, mut buff: &mut [u8]) -> Result<()> {
        while !buff.is_empty() {
            let need_bytes = buff.len();
            let ready_bytes = &self.buffer[self.next_byte..self.current_buffer_size];
            if ready_bytes.len() >= need_bytes {
                buff.clone_from_slice(&ready_bytes[..need_bytes]);
                self.next_byte += need_bytes;
                break;
            } else {
                buff[..ready_bytes.len()].clone_from_slice(ready_bytes);
                buff = &mut buff[ready_bytes.len()..];
                self.next_byte = 0;
                self.generate_one_batch(aes)?;
            }
        }
        Ok(())
    }

    fn recursively_generate_value(&mut self, aes: &mut Crypter, tp: Type) -> Result<Value> {
        match tp {
            Type::Scalar(_) | Type::Array(_, _) => {
                let bit_size = get_size_in_bits(tp)?;
                let byte_size = (bit_size + 7) / 8;
                // the last random byte should contain bits_to_flush zeros
                let bits_to_flush = 8 * byte_size - bit_size;
                let mut bytes = self.generate_random_bytes(aes, byte_size)?;
                // Remove unused random bits
                if !bytes.is_empty() {
                    *bytes.last_mut().unwrap() >>= bits_to_flush;
                }
                Ok(Value::from_bytes(bytes))
            }
            Type::Tuple(_) | Type::Vector(_, _) | Type::NamedTuple(_) => {
                let ts = get_types_vector(tp)?;
                let mut v = vec![];
                for sub_t in ts {
                    let value = self.recursively_generate_value(aes, (*sub_t).clone())?;
                    v.push(value);
                }
                Ok(Value::from_vector(v))
            }
        }
    }

    // Generates a random number from 0..2^(8 * NEED_BYTES).
    fn generate_random_number_const<const NEED_BYTES: usize>(
        &mut self,
        aes: &mut Crypter,
    ) -> Result<u64> {
        let mut res = [0u8; 8];
        // Note: sometimes we copy garbage bytes, but they are discarded later.
        res.copy_from_slice(&self.buffer[self.next_byte..self.next_byte + 8]);

        let use_bytes = std::cmp::min(self.current_buffer_size - self.next_byte, NEED_BYTES);
        if use_bytes == NEED_BYTES {
            self.next_byte += use_bytes;
        } else {
            self.generate_one_batch(aes)?;
            self.next_byte = NEED_BYTES - use_bytes;
            res[use_bytes..NEED_BYTES].copy_from_slice(&self.buffer[..self.next_byte]);
        }
        let mask = if NEED_BYTES == 8 {
            u64::MAX
        } else {
            (1 << (NEED_BYTES * 8)) - 1
        };
        Ok(u64::from_le_bytes(res) & mask)
    }

    // Generates a random number from 0..2^(8 * need_bytes).
    fn generate_random_number(&mut self, aes: &mut Crypter, need_bytes: usize) -> Result<u64> {
        match need_bytes {
            1 => self.generate_random_number_const::<1>(aes),
            2 => self.generate_random_number_const::<2>(aes),
            3 => self.generate_random_number_const::<3>(aes),
            4 => self.generate_random_number_const::<4>(aes),
            5 => self.generate_random_number_const::<5>(aes),
            6 => self.generate_random_number_const::<6>(aes),
            7 => self.generate_random_number_const::<7>(aes),
            8 => self.generate_random_number_const::<8>(aes),
            _ => Err(runtime_error!("Unsupported need bytes")),
        }
    }

    fn generate_u32_in_range(&mut self, aes: &mut Crypter, modulus: u32) -> Result<u32> {
        let modulus = modulus as u64;
        // Generate one extra byte of randomness to have reasonably low resampling probability.
        let need_bytes = (modulus.next_power_of_two().trailing_zeros() + 7) / 8 + 1;
        // The maximum possible value we could generate with `need_bytes` random bytes.
        // need_bytes <= 5 because modulus is below 2^32.
        let max_rand_value = (1u64 << (need_bytes as u64 * 8)) - 1;
        let num_biased = (max_rand_value + 1) % modulus;
        let rejection_bound = max_rand_value - num_biased;
        loop {
            let rand_value = self.generate_random_number(aes, need_bytes as usize)?;
            if rand_value <= rejection_bound {
                return Ok((rand_value % modulus) as u32);
            }
        }
    }
}

// Basic entropy test.
// The plugin estimator is computed and compared to the expected entropy
// of uniform distribution, i.e. 8 as we output bytes.
// The estimation error, abs(entropy - 8), can be heuristically bounded
// with overwhelming probability by 4*(d-1)/n with d = 256 in our case
// (see Theorem 1 in https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/lec5.pdf
// and On a statistical estimate for the entropy of a sequence of independent random variables by
// Basharin, GP (1959);
// note that sigma = 0 for any uniform distribution).
// Use this test to check uniformity of bytes.
// In general case, it is too strict.
pub fn entropy_test(counters: [u32; 256], n: u64) -> bool {
    let mut entropy = 0f64;
    for c in counters {
        let prob_c = (c as f64) / (n as f64);
        entropy -= prob_c.log2() * prob_c;
    }
    let precision = (1020_f64) / (n as f64);
    (entropy - 8f64).abs() < precision
}

// Computes chi squared statistics
pub fn chi_statistics(counters: &[u64], expected_count_per_element: u64) -> f64 {
    let mut chi_statistics = 0_f64;
    for c in counters {
        chi_statistics += (*c as f64 - expected_count_per_element as f64).powi(2);
    }
    chi_statistics / expected_count_per_element as f64
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::data_types::{
        array_type, named_tuple_type, scalar_type, tuple_type, vector_type, BIT, INT32, UINT64,
        UINT8,
    };

    #[test]

    fn test_prng_fixed_seed() {
        let helper = |n: usize| -> Result<()> {
            let seed = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F";
            let mut prng1 = PRNG::new(Some(seed.clone()))?;
            let mut prng2 = PRNG::new(Some(seed.clone()))?;
            let rand_bytes1 = prng1.get_random_bytes(n)?;
            let rand_bytes2 = prng2.get_random_bytes(n)?;
            assert_eq!(rand_bytes1, rand_bytes2);
            Ok(())
        };
        helper(1).unwrap();
        helper(19).unwrap();
        helper(1000).unwrap();
    }

    #[test]
    fn test_prng_random_seed() {
        let mut prng = PRNG::new(None).unwrap();
        let mut counters = [0; 256];
        let n = 10_000_001;
        let rand_bytes = prng.get_random_bytes(n).unwrap();
        for byte in rand_bytes {
            counters[byte as usize] += 1;
        }

        assert!(entropy_test(counters, n as u64));
    }

    #[test]
    fn test_prng_random_value() {
        let mut g = PRNG::new(None).unwrap();
        let mut helper = |t: Type| -> Result<()> {
            let v = g.get_random_value(t.clone())?;
            assert!(v.check_type(t)?);
            Ok(())
        };
        || -> Result<()> {
            helper(scalar_type(BIT))?;
            helper(scalar_type(UINT8))?;
            helper(scalar_type(INT32))?;
            helper(array_type(vec![2, 5], BIT))?;
            helper(array_type(vec![2, 5], UINT8))?;
            helper(array_type(vec![2, 5], INT32))?;
            helper(tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]))?;
            helper(tuple_type(vec![
                vector_type(3, scalar_type(BIT)),
                vector_type(5, scalar_type(BIT)),
                scalar_type(BIT),
                scalar_type(INT32),
            ]))?;
            helper(named_tuple_type(vec![
                ("field 1".to_owned(), scalar_type(BIT)),
                ("field 2".to_owned(), scalar_type(INT32)),
            ]))
        }()
        .unwrap()
    }
    #[test]
    fn test_prng_random_value_flush() {
        let mut g = PRNG::new(None).unwrap();
        let mut helper = |t: Type, expected: u8| -> Result<()> {
            let v = g.get_random_value(t.clone())?;
            v.access_bytes(|bytes| {
                if !bytes.is_empty() {
                    assert!(bytes.last() < Some(&expected));
                }
                Ok(())
            })?;
            Ok(())
        };
        || -> Result<()> {
            helper(array_type(vec![2, 5], BIT), 4)?;
            helper(array_type(vec![3, 3], BIT), 2)?;
            helper(array_type(vec![7], BIT), 128)?;
            helper(scalar_type(BIT), 2)
        }()
        .unwrap();
    }
    #[test]
    fn test_prng_random_u64_modulo() {
        || -> Result<()> {
            let mut g = PRNG::new(None).unwrap();

            let m = 100_u64;
            let mut counters = vec![0; m as usize];
            let expected_count_per_int = 1000;
            let n = expected_count_per_int * m;
            for _ in 0..n {
                let r = g.get_random_in_range(Some(m))?;
                counters[r as usize] += 1;
            }

            // Chi-square test with significance level 10^(-6)
            // <https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm>
            // Critical value is computed with m-1 degrees of freedom
            let chi2 = chi_statistics(&counters, expected_count_per_int);
            assert!(chi2 < 180.792_f64);

            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_prf_fixed_key() {
        || -> Result<()> {
            let key = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F";
            let mut prf1 = Prf::new(Some(key.clone()))?;
            let mut prf2 = Prf::new(Some(key.clone()))?;
            for i in 0..100_000u64 {
                assert_eq!(prf1.output_bytes(i, 1)?, prf2.output_bytes(i, 1)?);
                assert_eq!(prf1.output_bytes(i, 5)?, prf2.output_bytes(i, 5)?);
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_prf_random_key() {
        // basic entropy test
        || -> Result<()> {
            let mut prf = Prf::new(None)?;
            let mut counters = [0; 256];
            let n = 100_000u64;
            let k = 10u64;
            for i in 0..n {
                let out = prf.output_bytes(i, k)?;
                for byte in out {
                    counters[byte as usize] += 1;
                }
            }
            assert!(entropy_test(counters, n * k as u64));
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_prf_output_value() {
        let mut g = Prf::new(None).unwrap();
        let mut helper = |t: Type| -> Result<()> {
            let v1 = g.output_value(15, t.clone())?;
            let v2 = g.output_value(15, t.clone())?;
            assert!(v1.check_type(t.clone())?);
            assert!(v2.check_type(t.clone())?);
            assert_eq!(v1, v2);
            if let Type::Tuple(_) | Type::Vector(_, _) | Type::NamedTuple(_) = t.clone() {
                let values = v1.to_vector()?;
                // check that not all values are equal
                let mut all_equal = true;
                for i in 1..values.len() {
                    all_equal &= values[i - 1] == values[i];
                }
                assert!(!all_equal);

                // checks that all scalars contained in values are different
                // this test passes with overwhelming probability only for UINT64 scalars
                let mut numbers = vec![];
                let types = get_types_vector(t)?;
                for i in 0..types.len() {
                    let tp = (*types[i]).clone();
                    if !tp.is_array() {
                        return Ok(());
                    }
                    if tp.get_scalar_type() != UINT64 {
                        return Ok(());
                    }
                    let mut tmp = values[i].to_flattened_array_u64(tp)?;
                    numbers.append(&mut tmp)
                }
                let mut tmp_numbers = numbers.clone();
                tmp_numbers.sort_unstable();
                tmp_numbers.dedup();
                assert_eq!(tmp_numbers.len(), numbers.len());
            }
            Ok(())
        };
        || -> Result<()> {
            helper(scalar_type(BIT))?;
            helper(scalar_type(UINT8))?;
            helper(scalar_type(INT32))?;
            helper(array_type(vec![3, 4], BIT))?;
            helper(array_type(vec![4, 2], UINT8))?;
            helper(array_type(vec![6, 2], INT32))?;
            helper(tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]))?;
            helper(tuple_type(vec![
                vector_type(3, scalar_type(BIT)),
                vector_type(5, scalar_type(BIT)),
                scalar_type(BIT),
                scalar_type(INT32),
            ]))?;
            helper(tuple_type(vec![
                scalar_type(INT32),
                scalar_type(INT32),
                scalar_type(INT32),
                scalar_type(INT32),
            ]))?;
            helper(tuple_type(vec![
                array_type(vec![2, 2], INT32),
                array_type(vec![2, 2], INT32),
                array_type(vec![2, 2], INT32),
                array_type(vec![2, 2], INT32),
            ]))?;
            helper(tuple_type(vec![
                array_type(vec![2, 1, 2], UINT64),
                array_type(vec![2, 3, 2], UINT64),
                array_type(vec![2, 2, 1], UINT64),
                array_type(vec![3, 3, 2], UINT64),
            ]))?;
            helper(named_tuple_type(vec![
                ("field 1".to_owned(), scalar_type(BIT)),
                ("field 2".to_owned(), scalar_type(INT32)),
            ]))
        }()
        .unwrap();

        let mut helper_flush = |t: Type, expected: u8| -> Result<()> {
            let v = g.output_value(181, t.clone())?;
            v.access_bytes(|bytes| {
                if !bytes.is_empty() {
                    assert!(bytes.last() < Some(&expected));
                }
                Ok(())
            })?;
            Ok(())
        };
        || -> Result<()> {
            helper_flush(array_type(vec![1, 5], BIT), 32)?;
            helper_flush(array_type(vec![3, 3, 3], BIT), 8)?;
            helper_flush(array_type(vec![2, 6], BIT), 16)?;
            helper_flush(scalar_type(BIT), 2)
        }()
        .unwrap();
    }

    #[test]
    fn test_generate_u32_in_range() -> Result<()> {
        let mut prf = Prf::new(None)?;
        // critical_value[i] is a precomputed critical value for the Chi-square test with
        // i degrees of freedom and significance level 10^(-6).
        let critical_value = [0f64, 23.9281, 27.6310, 30.6648, 33.3768, 35.8882];
        for n in 2..6 {
            let mut session = PrfSession::new(0, 1)?;
            let expected_count = 1000000;
            let runs = n * expected_count;
            let mut stats: HashMap<u32, u64> = HashMap::new();
            for _ in 0..runs {
                let x = session.generate_u32_in_range(&mut prf.aes, n)?;
                assert!(x < n);
                *stats.entry(x).or_default() += 1;
            }
            let counters: Vec<u64> = stats.values().cloned().collect();
            let chi2 = chi_statistics(&counters, expected_count as u64);
            assert!(chi2 < critical_value[(n - 1) as usize]);
        }
        Ok(())
    }

    #[test]
    fn test_prf_output_permutation() -> Result<()> {
        let mut prf = Prf::new(None)?;
        let mut helper = |n: u64| -> Result<()> {
            let result_type = array_type(vec![n], UINT64);
            let mut perm_statistics: HashMap<Vec<u64>, u64> = HashMap::new();
            let expected_count_per_perm = 100;
            let n_factorial: u64 = (2..=n).product();
            let runs = expected_count_per_perm * n_factorial;
            for input in 0..runs {
                let result_value = prf.output_permutation(input, n)?;
                let perm = result_value.to_flattened_array_u64(result_type.clone())?;

                let mut perm_sorted = perm.clone();
                perm_sorted.sort();
                let range_vec: Vec<u64> = (0..n).collect();
                assert_eq!(perm_sorted, range_vec);

                perm_statistics
                    .entry(perm)
                    .and_modify(|counter| *counter += 1)
                    .or_insert(0);
            }

            // Check that all permutations occurred in the experiments
            assert_eq!(perm_statistics.len() as u64, n_factorial);

            // Chi-square test with significance level 10^(-6)
            // <https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm>
            if n > 1 {
                let counters: Vec<u64> = perm_statistics.values().map(|c| *c).collect();
                let chi2 = chi_statistics(&counters, expected_count_per_perm);
                // Critical value is computed with n!-1 degrees of freedom
                if n == 4 {
                    assert!(chi2 < 70.5496_f64);
                }
                if n == 5 {
                    assert!(chi2 < 207.1986_f64);
                }
            }
            Ok(())
        };
        helper(1)?;
        helper(4)?;
        helper(5)
    }

    #[test]
    fn test_prf_output_permutation_correctness() -> Result<()> {
        let mut prf = Prf::new(None)?;
        let mut helper = |n: u64| -> Result<()> {
            let result_type = array_type(vec![n], UINT64);
            let result_value = prf.output_permutation(0, n)?;
            let perm = result_value.to_flattened_array_u64(result_type.clone())?;

            let mut perm_sorted = perm.clone();
            perm_sorted.sort();
            let range_vec: Vec<u64> = (0..n).collect();
            assert_eq!(perm_sorted, range_vec);
            Ok(())
        };
        helper(1)?;
        helper(10)?;
        helper(100)?;
        helper(1000)?;
        helper(10000)?;
        helper(100000)?;
        helper(1000000)?;
        Ok(())
    }
}
