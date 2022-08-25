use crate::data_types::{get_size_in_bits, get_types_vector, Type};
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

/// Cryptographic pseudo-random generator based on AES-128 in the counter mode.
/// If the seed is private, the security is based on the key-recovery hardness assumption of AES
/// and [the PRP/PRF(Prf) switching lemma](https://eprint.iacr.org/2004/331.pdf).
/// Less than 2^64 128-bit strings should be sampled to avoid distinguishing attacks due to birthday paradox.
/// The empirical randomness properties of AES output is shown in "Empirical Evidence Concerning AES"
/// by Peter Hellekalek and Stefan Wegenkittl.
pub struct PRNG {
    counter: u128,
    random_bytes: Vec<u8>,
    aes: Crypter,
}
/// The following implementation is not thread-safe as several copies of PRNG
/// can concurrently access the system random generator
impl PRNG {
    pub fn new(seed: Option<[u8; SEED_SIZE]>) -> Result<PRNG> {
        let err = |_| runtime_error!("Crypter didn't initialize");
        match seed {
            Some(bytes) => {
                let mut c = Crypter::new(Cipher::aes_128_ecb(), Mode::Encrypt, &bytes, None)
                    .map_err(err)?;
                c.pad(false);
                Ok(PRNG {
                    counter: 0u128,
                    random_bytes: vec![],
                    aes: c,
                })
            }
            None => {
                let mut bytes = [0u8; SEED_SIZE];
                get_bytes_from_os(&mut bytes)?;
                let mut c = Crypter::new(Cipher::aes_128_ecb(), Mode::Encrypt, &bytes, None)
                    .map_err(err)?;
                c.pad(false);
                Ok(PRNG {
                    counter: 0u128,
                    random_bytes: vec![],
                    aes: c,
                })
            }
        }
    }

    fn refill_random(&mut self) -> Result<()> {
        let counter_bytes = self.counter.to_le_bytes();
        // additional block is needed to perform encryption,
        // check here https://www.openssl.org/docs/manmaster/man3/EVP_CipherUpdate.html
        let mut res = vec![0; 2 * SEED_SIZE];
        let count = self
            .aes
            .update(&counter_bytes, &mut res)
            .map_err(|_| runtime_error!("Crypter didn't manage to update"))?;
        // finalization of Crypter is unnecessary since padding is turned off
        // check here https://www.openssl.org/docs/manmaster/man3/EVP_CipherUpdate.html
        if count != SEED_SIZE {
            return Err(runtime_error!(
                "AES encryption returned a wrong number of bytes"
            ));
        }
        res.truncate(count);
        self.random_bytes = res;
        self.counter += 1;
        Ok(())
    }

    pub fn get_random_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut res = vec![];
        while res.len() < n {
            let num_to_fill = n - res.len();
            if num_to_fill >= self.random_bytes.len() {
                res.append(&mut self.random_bytes);
                self.refill_random()?;
            } else {
                let l = self.random_bytes.len();
                let mut tmp_vec: Vec<u8> = self.random_bytes.drain(l - num_to_fill..l).collect();
                res.append(&mut tmp_vec)
            }
        }
        Ok(res)
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
    out_vec: Vec<u8>,
}

impl Prf {
    pub fn new(key: Option<[u8; SEED_SIZE]>) -> Result<Prf> {
        let err = |_| runtime_error!("Crypter didn't initialize");
        match key {
            Some(bytes) => {
                let mut c = Crypter::new(Cipher::aes_128_ecb(), Mode::Encrypt, &bytes, None)
                    .map_err(err)?;
                c.pad(false);
                Ok(Prf {
                    aes: c,
                    out_vec: vec![0u8; 2 * SEED_SIZE],
                })
            }
            None => {
                let mut gen = PRNG::new(None)?;
                let key_bytes = gen.get_random_key()?;
                let mut c = Crypter::new(Cipher::aes_128_ecb(), Mode::Encrypt, &key_bytes, None)
                    .map_err(err)?;
                c.pad(false);
                Ok(Prf {
                    aes: c,
                    out_vec: vec![0u8; 2 * SEED_SIZE],
                })
            }
        }
    }

    fn generate_one_batch(&mut self, input: u128) -> Result<()> {
        let i_bytes = input.to_le_bytes();
        let count = self
            .aes
            .update(&i_bytes, &mut self.out_vec)
            .map_err(|_| runtime_error!("Crypter didn't manage to update"))?;
        // finalization of Crypter is unnecessary since padding is turned off
        // check here https://www.openssl.org/docs/manmaster/man3/EVP_CipherUpdate.html
        if count != SEED_SIZE {
            return Err(runtime_error!(
                "AES encryption returned a wrong number of bytes"
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn output_bytes(&mut self, input: u64, n: u64) -> Result<Vec<u8>> {
        let mut res = vec![];
        let ext_input = (input as u128) << 64;
        let calls = (n - 1) / SEED_SIZE as u64 + 1;
        let rem = n - (calls - 1) * SEED_SIZE as u64;
        for i in 0..calls as u128 {
            let ext_i = ext_input + i;
            self.generate_one_batch(ext_i)?;
            if i == calls as u128 - 1 {
                res.extend(&self.out_vec[0..rem as usize]);
            } else {
                res.extend(&self.out_vec[0..SEED_SIZE]);
            }
        }
        Ok(res)
    }

    fn recursively_generate_value(&mut self, iv: u128, tp: Type) -> Result<(Value, u128)> {
        match tp {
            Type::Scalar(_) | Type::Array(_, _) => {
                let bit_size = get_size_in_bits(tp)?;
                let byte_size = (bit_size + 7) / 8;
                // the last random byte should contain bits_to_flush zeros
                let bits_to_flush = 8 * byte_size - bit_size;
                let mut bytes = vec![];
                let calls = (byte_size - 1) / SEED_SIZE as u64 + 1;
                let rem = byte_size - (calls - 1) * SEED_SIZE as u64;
                for i in 0..calls as u128 {
                    let ext_iv = iv + i;
                    self.generate_one_batch(ext_iv)?;
                    if i == calls as u128 - 1 {
                        bytes.extend(&self.out_vec[0..rem as usize]);
                    } else {
                        bytes.extend(&self.out_vec[0..SEED_SIZE]);
                    }
                }
                // Remove unused random bits
                if !bytes.is_empty() {
                    *bytes.last_mut().unwrap() >>= bits_to_flush;
                }
                Ok((Value::from_bytes(bytes), iv + calls as u128))
            }
            Type::Tuple(_) | Type::Vector(_, _) | Type::NamedTuple(_) => {
                let ts = get_types_vector(tp)?;
                let mut v = vec![];
                let mut ext_iv = iv;
                for sub_t in ts {
                    let (value, next_iv) =
                        self.recursively_generate_value(ext_iv, (*sub_t).clone())?;
                    v.push(value);
                    ext_iv = next_iv;
                }
                Ok((Value::from_vector(v), ext_iv))
            }
        }
    }

    pub(super) fn output_value(&mut self, input: u64, t: Type) -> Result<Value> {
        let ext_input = (input as u128) << 64;
        let value = self.recursively_generate_value(ext_input, t)?.0;
        Ok(value)
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
pub fn entropy_test(counters: [u32; 256], n: u64) -> bool {
    let mut entropy = 0f64;
    for c in counters {
        let prob_c = (c as f64) / (n as f64);
        entropy -= prob_c.log2() * prob_c;
    }
    let precision = (1020_f64) / (n as f64);
    (entropy - 8f64).abs() < precision
}

#[cfg(test)]
mod tests {
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
}
