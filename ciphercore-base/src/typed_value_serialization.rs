//! This module implements serialization/deserialization of TypedValue for the human-readable formats.
//! If the format is non-human-readable there would be used default (provided by rust) serialization.
use std::fmt::{self, Debug};

use crate::data_types::{
    ArrayShape, ScalarType, Type, BIT, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8,
};
use crate::data_values::Value;
use crate::typed_value::TypedValue;
use crate::typed_value_operations::{
    FromVectorMode, TypedValueArrayOperations, TypedValueOperations,
};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{de, ser};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::result::Result;

impl Serialize for TypedValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            self.serialize_human_readable(serializer)
        } else {
            (&self.t, &self.value).serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for TypedValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<TypedValue, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            TypedValue::deserialize_human_readable(deserializer)
        } else {
            let tv = <(Type, Value)>::deserialize(deserializer)?;
            Ok(TypedValue {
                t: tv.0,
                value: tv.1,
                name: None,
            })
        }
    }
}

macro_rules! ser_value_to_scalar_aux {
    ($s:expr, $v:expr, $t:expr, $cnv:ident) => {
        $s.serialize_field("value", &$v.$cnv($t.clone()).map_err(ser::Error::custom)?)
    };
}

macro_rules! ser_value_to_scalar_array_aux {
    ($s:expr, $v:expr, $t:expr, $cnv:ident, $shape:expr) => {
        $s.serialize_field(
            "value",
            &ShapedArray {
                array: $v.$cnv($t.clone()).map_err(ser::Error::custom)?,
                shape: $shape,
            },
        )
    };
}

macro_rules! unpack_serialized_data_model {
    ($v:expr, $t:ident, $x:ident) => {
        match $v {
            SerializedDataModel::$t($x) => Ok($x),
            _ => Err(de::Error::custom("Unable to unpack enum!")),
        }
    };
}

#[derive(Debug, Serialize, Deserialize)]
struct NamedTypedValue {
    name: String,
    value: TypedValue,
}

#[derive(Debug)]
struct ShapedArray<T> {
    array: Vec<T>,
    shape: ArrayShape,
}

impl<T: Clone> ShapedArray<T> {
    fn to_ndarray(&self) -> Result<ndarray::ArrayD<T>, ndarray::ShapeError> {
        let shape: Vec<usize> = self.shape.iter().map(|x| *x as usize).collect();
        let ndarr = ndarray::Array::from_vec(self.array.clone());
        ndarr.into_shape(shape)
    }
}

#[derive(Debug)]
enum SerializedDataModel {
    Array(ShapedArray<u64>),
    Vector(Vec<TypedValue>),
    Value(TypedValue),
    NamedTuple(Vec<(String, TypedValue)>),
}

impl<T: Clone + Serialize> Serialize for ShapedArray<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Clone + Serialize,
    {
        match self.shape.len() {
            0 => Err(ser::Error::custom("Shape can not be empty")),
            1 => self.array.serialize(serializer),
            _ => {
                let len = self.shape[0] as usize;
                let new_shape: Vec<u64> = self.shape[1..].to_vec();
                let mut s = serializer.serialize_seq(Some(len))?;
                if self.array.len() % len != 0 {
                    return Err(ser::Error::custom("Array shape mismatch"));
                }
                let chunk_size = self.array.len() / len;
                for chunk in self.array.chunks(chunk_size) {
                    s.serialize_element(&ShapedArray {
                        array: chunk.to_vec().clone(),
                        shape: new_shape.clone(),
                    })?;
                }
                s.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for SerializedDataModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(SerializedDataModelVisitor)
    }
}

struct SerializedDataModelVisitor;

impl<'de> Visitor<'de> for SerializedDataModelVisitor {
    type Value = SerializedDataModel;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Human-readable serialized TypedValue")
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(SerializedDataModel::Array(ShapedArray {
            array: vec![value as u64],
            shape: vec![],
        }))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(SerializedDataModel::Array(ShapedArray {
            array: vec![value],
            shape: vec![],
        }))
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(SerializedDataModel::Array(ShapedArray {
            array: vec![value as u64],
            shape: vec![],
        }))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Kind,
            Name,
            Type,
            Value,
        }

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Kind {
            Scalar,
            Array,
            Vector,
            Tuple,
            #[serde(rename(deserialize = "named tuple"))]
            NamedTuple,
        }

        let mut kind: Option<Kind> = None;
        let mut t: Option<String> = None;
        let mut name: Option<String> = None;
        let mut opt_value: Option<SerializedDataModel> = None;
        while let Some(key) = map.next_key()? {
            match key {
                Field::Kind => {
                    if kind.is_some() {
                        return Err(de::Error::duplicate_field("kind"));
                    }
                    kind = Some(map.next_value()?);
                }
                Field::Type => {
                    if t.is_some() {
                        return Err(de::Error::duplicate_field("type"));
                    }
                    t = Some(map.next_value()?);
                }
                Field::Value => {
                    if opt_value.is_some() {
                        return Err(de::Error::duplicate_field("value"));
                    }
                    opt_value = Some(map.next_value()?);
                }
                Field::Name => {
                    if name.is_some() {
                        return Err(de::Error::duplicate_field("name"));
                    }
                    name = Some(map.next_value()?);
                }
            }
        }
        let value = opt_value.ok_or_else(|| de::Error::missing_field("value"))?;
        if let Some(n) = name {
            if kind.is_some() || t.is_some() {
                return Err(de::Error::custom(
                    "Unexpected field: \"name\" can not be together with \"type\" or \"kind\".",
                ));
            }
            if let SerializedDataModel::Value(tv) = value {
                return Ok(SerializedDataModel::NamedTuple(vec![(n, tv)]));
            } else {
                return Err(de::Error::custom(
                    "Unexpected field: \"name\" can not be together with \"type\" or \"kind\".",
                ));
            }
        }

        let kind = kind.ok_or_else(|| de::Error::missing_field("kind"))?;

        match kind {
            Kind::Scalar => {
                let st = t
                    .ok_or_else(|| de::Error::missing_field("t"))?
                    .parse::<ScalarType>()
                    .map_err(de::Error::custom)?;
                if let SerializedDataModel::Array(a) = value {
                    if a.array.len() != 1 {
                        Err(de::Error::custom(
                            "The value doesn't match to kind \"scalar\".",
                        ))
                    } else {
                        Ok(SerializedDataModel::Value(
                            TypedValue::from_scalar(a.array[0], st).map_err(de::Error::custom)?,
                        ))
                    }
                } else {
                    Err(de::Error::custom(
                        "The value doesn't match to kind \"scalar\".",
                    ))
                }
            }
            Kind::Array => {
                let st = t
                    .ok_or_else(|| de::Error::missing_field("t"))?
                    .parse::<ScalarType>()
                    .map_err(de::Error::custom)?;
                if let SerializedDataModel::Array(a) = value {
                    Ok(SerializedDataModel::Value(
                        TypedValue::from_ndarray(a.to_ndarray().map_err(de::Error::custom)?, st)
                            .map_err(de::Error::custom)?,
                    ))
                } else {
                    Err(de::Error::custom(
                        "The value doesn't match to kind \"scalar\".",
                    ))
                }
            }
            Kind::Vector => {
                if let SerializedDataModel::Vector(v) = value {
                    Ok(SerializedDataModel::Value(
                        TypedValue::from_vector(v, FromVectorMode::Vector)
                            .map_err(de::Error::custom)?,
                    ))
                } else {
                    Err(de::Error::custom(
                        "The value doesn't match to kind \"vector\".",
                    ))
                }
            }
            Kind::Tuple => {
                if let SerializedDataModel::Vector(v) = value {
                    Ok(SerializedDataModel::Value(
                        TypedValue::from_vector(v, FromVectorMode::Tuple)
                            .map_err(de::Error::custom)?,
                    ))
                } else {
                    Err(de::Error::custom(
                        "The value doesn't match to kind \"tuple\".",
                    ))
                }
            }
            Kind::NamedTuple => {
                if let SerializedDataModel::NamedTuple(v) = value {
                    let new_v: Vec<TypedValue> = v
                        .iter()
                        .map(|v| {
                            TypedValue::new_named(v.1.t.clone(), v.1.value.clone(), v.0.clone())
                                .map_err(de::Error::custom)
                        })
                        .collect::<Result<Vec<TypedValue>, A::Error>>()?;
                    Ok(SerializedDataModel::Value(
                        TypedValue::from_vector(new_v, FromVectorMode::Tuple)
                            .map_err(de::Error::custom)?,
                    ))
                } else {
                    Err(de::Error::custom(
                        "The value doesn't match to kind \"named tuple\".",
                    ))
                }
            }
        }
    }

    /// Processing a sequence. This method expects that all processed elements of the sequence
    /// expected to have the same enum variants.
    fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let data: Vec<Self::Value> =
            serde::Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))?;
        if data.is_empty() {
            return Ok(SerializedDataModel::Vector(vec![]));
        }
        for elem in &data {
            if std::mem::discriminant(elem) != std::mem::discriminant(&data[0]) {
                return Err(de::Error::custom(
                    "Inconsistent sequence: expected all elements have the same enum variant.",
                ));
            }
        }
        match data[0] {
            SerializedDataModel::Array(_) => {
                let mut v = vec![];
                let mut updated_shape = false;
                let mut new_shape = vec![data.len() as u64];
                for item in data {
                    let sub_arr = unpack_serialized_data_model!(item, Array, sub_arr)?;
                    if !updated_shape {
                        new_shape.extend(sub_arr.shape);
                        updated_shape = true;
                    }
                    v.extend(sub_arr.array);
                }
                Ok(SerializedDataModel::Array(ShapedArray {
                    array: v,
                    shape: new_shape,
                }))
            }
            SerializedDataModel::Vector(_) => Err(de::Error::custom(
                "Found Vector of vectors of TypedValue which is unsupported.",
            )),
            SerializedDataModel::NamedTuple(_) => {
                let mut v = vec![];
                for item in data {
                    let nv = unpack_serialized_data_model!(item, NamedTuple, nv)?;
                    v.extend(nv);
                }
                Ok(SerializedDataModel::NamedTuple(v))
            }
            SerializedDataModel::Value(_) => {
                let mut v = vec![];
                for item in data {
                    let tv = unpack_serialized_data_model!(item, Value, tv)?;
                    v.push(tv);
                }
                Ok(SerializedDataModel::Vector(v))
            }
        }
    }
}

impl TypedValue {
    fn serialize_human_readable<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("TypedValue", 3)?;
        match &self.t {
            Type::Scalar(st) => {
                s.serialize_field("kind", &"scalar")?;
                s.serialize_field("type", &format!("{}", st))?;
                match *st {
                    BIT => ser_value_to_scalar_aux!(s, self.value, st, to_u8)?,
                    UINT8 => ser_value_to_scalar_aux!(s, self.value, st, to_u8)?,
                    INT8 => ser_value_to_scalar_aux!(s, self.value, st, to_i8)?,
                    UINT16 => ser_value_to_scalar_aux!(s, self.value, st, to_u16)?,
                    INT16 => ser_value_to_scalar_aux!(s, self.value, st, to_i16)?,
                    UINT32 => ser_value_to_scalar_aux!(s, self.value, st, to_u32)?,
                    INT32 => ser_value_to_scalar_aux!(s, self.value, st, to_i32)?,
                    UINT64 => ser_value_to_scalar_aux!(s, self.value, st, to_u64)?,
                    INT64 => ser_value_to_scalar_aux!(s, self.value, st, to_i64)?,
                    _ => {
                        return Err(ser::Error::custom("Invalid scalar type"));
                    }
                };
                s.end()
            }
            Type::Array(ref_shape, st) => {
                s.serialize_field("kind", &"array")?;
                s.serialize_field("type", &format!("{}", st))?;
                let shape = ref_shape.clone();
                match *st {
                    BIT => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_u8,
                        shape
                    )?,
                    UINT8 => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_u8,
                        shape
                    )?,
                    INT8 => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_i8,
                        shape
                    )?,
                    UINT16 => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_u16,
                        shape
                    )?,
                    INT16 => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_i16,
                        shape
                    )?,
                    UINT32 => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_u32,
                        shape
                    )?,
                    INT32 => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_i32,
                        shape
                    )?,
                    UINT64 => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_u64,
                        shape
                    )?,
                    INT64 => ser_value_to_scalar_array_aux!(
                        s,
                        self.value,
                        self.t,
                        to_flattened_array_i64,
                        shape
                    )?,
                    _ => {
                        return Err(ser::Error::custom("Invalid scalar type"));
                    }
                };
                s.end()
            }
            Type::Tuple(_) => {
                s.serialize_field("kind", "tuple")?;
                s.skip_field("type")?;
                let result = self.to_vector().map_err(ser::Error::custom)?;
                s.serialize_field("value", &result)?;
                s.end()
            }
            Type::NamedTuple(pairs) => {
                s.serialize_field("kind", "named tuple")?;
                s.skip_field("type")?;
                let sub_values = self.to_vector().map_err(ser::Error::custom)?;
                let mut result = vec![];
                for (n_t, v) in pairs.iter().zip(sub_values.iter()) {
                    result.push(NamedTypedValue {
                        name: n_t.0.clone(),
                        value: v.clone(),
                    });
                }
                s.serialize_field("value", &result)?;
                s.end()
            }
            Type::Vector(_, _) => {
                s.serialize_field("kind", "vector")?;
                s.skip_field("type")?;
                let result = self.to_vector().map_err(ser::Error::custom)?;
                s.serialize_field("value", &result)?;
                s.end()
            }
        }
    }
}

impl<'de> TypedValue {
    fn deserialize_human_readable<D>(deserializer: D) -> std::result::Result<TypedValue, D::Error>
    where
        D: Deserializer<'de>,
    {
        let dm = SerializedDataModel::deserialize(deserializer)?;
        if let SerializedDataModel::Value(tv) = dm {
            Ok(tv)
        } else {
            Err(de::Error::custom("Not a Typed Value."))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{array_type, named_tuple_type, scalar_type, tuple_type, vector_type};
    use crate::errors::Result;

    fn test_scalar_or_array_helper(s: &str, t: Type, bytes: &[u8]) -> Result<()> {
        let tv = serde_json::from_str::<TypedValue>(s)?;
        assert_eq!(tv.t, t);
        tv.value.access_bytes(|b| {
            assert_eq!(b, bytes);
            Ok(())
        })?;
        assert_eq!(serde_json::to_string(&tv)?, s);
        Ok(())
    }

    fn test_scalar_helper(s: &str, st: ScalarType, bytes: &[u8]) -> Result<()> {
        test_scalar_or_array_helper(s, scalar_type(st), bytes)
    }

    fn test_array_helper(s: &str, st: ScalarType, bytes: &[u8]) -> Result<()> {
        test_scalar_or_array_helper(s, array_type(vec![1], st), bytes)
    }

    #[test]
    fn test_scalar_b() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"b","value":0}"#;
        test_scalar_helper(s, BIT, &[0])
    }
    #[test]
    fn test_array_b() -> Result<()> {
        let s = r#"{"kind":"array","type":"b","value":[0]}"#;
        test_array_helper(s, BIT, &[0])
    }
    #[test]
    fn test_scalar_u8() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"u8","value":123}"#;
        test_scalar_helper(s, UINT8, &[123])
    }
    #[test]
    fn test_array_u8() -> Result<()> {
        let s = r#"{"kind":"array","type":"u8","value":[123]}"#;
        test_array_helper(s, UINT8, &[123])
    }
    #[test]
    fn test_scalar_i8() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"i8","value":-123}"#;
        test_scalar_helper(s, INT8, &[133])
    }
    #[test]
    fn test_array_i8() -> Result<()> {
        let s = r#"{"kind":"array","type":"i8","value":[-123]}"#;
        test_array_helper(s, INT8, &[133])
    }
    #[test]
    fn test_scalar_u16() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"u16","value":1234}"#;
        test_scalar_helper(s, UINT16, &[210, 4])
    }
    #[test]
    fn test_array_u16() -> Result<()> {
        let s = r#"{"kind":"array","type":"u16","value":[1234]}"#;
        test_array_helper(s, UINT16, &[210, 4])
    }
    #[test]
    fn test_scalar_i16() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"i16","value":-1234}"#;
        test_scalar_helper(s, INT16, &[46, 251])
    }
    #[test]
    fn test_array_i16() -> Result<()> {
        let s = r#"{"kind":"array","type":"i16","value":[-1234]}"#;
        test_array_helper(s, INT16, &[46, 251])
    }
    #[test]
    fn test_scalar_u32() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"u32","value":123456}"#;
        test_scalar_helper(s, UINT32, &[64, 226, 1, 0])
    }
    #[test]
    fn test_array_u32() -> Result<()> {
        let s = r#"{"kind":"array","type":"u32","value":[123456]}"#;
        test_array_helper(s, UINT32, &[64, 226, 1, 0])
    }
    #[test]
    fn test_scalar_i32() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"i32","value":-123456}"#;
        test_scalar_helper(s, INT32, &[192, 29, 254, 255])
    }
    #[test]
    fn test_array_i32() -> Result<()> {
        let s = r#"{"kind":"array","type":"i32","value":[-123456]}"#;
        test_array_helper(s, INT32, &[192, 29, 254, 255])
    }
    #[test]
    fn test_scalar_u64() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"u64","value":123456}"#;
        test_scalar_helper(s, UINT64, &[64, 226, 1, 0, 0, 0, 0, 0])
    }
    #[test]
    fn test_array_u64() -> Result<()> {
        let s = r#"{"kind":"array","type":"u64","value":[123456]}"#;
        test_array_helper(s, UINT64, &[64, 226, 1, 0, 0, 0, 0, 0])
    }
    #[test]
    fn test_scalar_i64() -> Result<()> {
        let s = r#"{"kind":"scalar","type":"i64","value":-123456}"#;
        test_scalar_helper(s, INT64, &[192, 29, 254, 255, 255, 255, 255, 255])
    }
    #[test]
    fn test_array_i64() -> Result<()> {
        let s = r#"{"kind":"array","type":"i64","value":[-123456]}"#;
        test_array_helper(s, INT64, &[192, 29, 254, 255, 255, 255, 255, 255])
    }
    #[test]
    fn test_ndarray() -> Result<()> {
        let s = r#"{"kind":"array","type":"i32","value":[[-123456,123456],[13579,-13579]]}"#;
        let tv = serde_json::from_str::<TypedValue>(&s)?;
        assert_eq!(tv.t, array_type(vec![2, 2], INT32));
        tv.value.access_bytes(|b| {
            assert_eq!(
                b,
                &[192, 29, 254, 255, 64, 226, 1, 0, 11, 53, 0, 0, 245, 202, 255, 255]
            );
            Ok(())
        })?;
        assert_eq!(serde_json::to_string(&tv)?, s);
        Ok(())
    }
    #[test]
    fn test_tuple() -> Result<()> {
        let s = r#"{"kind":"tuple","value":[{"kind":"scalar","type":"i32","value":-123456},{"kind":"tuple","value":[]}]}"#;
        let tv = serde_json::from_str::<TypedValue>(&s)?;
        assert_eq!(
            tv.t,
            tuple_type(vec![scalar_type(INT32), tuple_type(vec![])])
        );
        tv.value.access_vector(|v| {
            assert_eq!(v.len(), 2);
            v[0].access_bytes(|b| {
                assert_eq!(b, &[192, 29, 254, 255]);
                Ok(())
            })?;
            v[1].access_vector(|v1| {
                assert!(v1.is_empty());
                Ok(())
            })?;
            Ok(())
        })?;
        assert_eq!(serde_json::to_string(&tv)?, s);
        Ok(())
    }
    #[test]
    fn test_named_tuple() -> Result<()> {
        let s = r#"{"kind":"named tuple","value":[{"name":"field1","value":{"kind":"scalar","type":"i32","value":-123456}},{"name":"field2","value":{"kind":"tuple","value":[]}}]}"#;
        let tv = serde_json::from_str::<TypedValue>(&s)?;
        assert_eq!(
            tv.t,
            named_tuple_type(vec![
                ("field1".to_owned(), scalar_type(INT32)),
                ("field2".to_owned(), tuple_type(vec![]))
            ])
        );
        tv.value.access_vector(|v| {
            assert_eq!(v.len(), 2);
            v[0].access_bytes(|b| {
                assert_eq!(b, &[192, 29, 254, 255]);
                Ok(())
            })?;
            v[1].access_vector(|v1| {
                assert!(v1.is_empty());
                Ok(())
            })?;
            Ok(())
        })?;
        assert_eq!(serde_json::to_string(&tv)?, s);
        Ok(())
    }
    #[test]
    fn test_vector() -> Result<()> {
        let s = r#"{"kind":"vector","value":[{"kind":"scalar","type":"i32","value":-123456},{"kind":"scalar","type":"i32","value":123456},{"kind":"scalar","type":"i32","value":-13579},{"kind":"scalar","type":"i32","value":13579}]}"#;
        let tv = serde_json::from_str::<TypedValue>(&s)?;
        assert_eq!(tv.t, vector_type(4, scalar_type(INT32)));
        tv.value.access_vector(|v| {
            assert_eq!(v.len(), 4);
            v[0].access_bytes(|b| {
                assert_eq!(b, &[192, 29, 254, 255]);
                Ok(())
            })?;
            v[1].access_bytes(|b| {
                assert_eq!(b, &[64, 226, 1, 0]);
                Ok(())
            })?;
            v[2].access_bytes(|b| {
                assert_eq!(b, &[245, 202, 255, 255]);
                Ok(())
            })?;
            v[3].access_bytes(|b| {
                assert_eq!(b, &[11, 53, 0, 0]);
                Ok(())
            })?;
            Ok(())
        })?;
        assert_eq!(serde_json::to_string(&tv)?, s);
        Ok(())
    }
    #[test]
    fn test_empty() -> Result<()> {
        let s = r#"{"kind":"vector","value":[]}"#;
        let tv = serde_json::from_str::<TypedValue>(&s)?;
        assert_eq!(tv.t, vector_type(0, tuple_type(vec![])));
        tv.value.access_vector(|v| {
            assert_eq!(v.len(), 0);
            Ok(())
        })?;
        assert_eq!(serde_json::to_string(&tv)?, s);
        Ok(())
    }
    #[test]
    fn test_err() -> Result<()> {
        let s = r#"[]"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":{}}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"wrongkind"}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"scalar","type":"i32","value":{}}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"scalar","type":"i32"}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"scalar","type":"bzz","value":123}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"scalar","type":[],"value":123}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"scalar","value":123}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"array","type":"i32","value":[[123,456],123]}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"tuple","value":{}}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"array","type":"i32","value":{}}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"vector","value":{}}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"vector","value":[{"kind":"scalar","type":"i32","value":123},{"kind":"tuple","value":{}}]}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"named tuple","value":{}}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"named tuple","value":[{}]}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"named tuple","value":[{"name":123,"value":{}}]}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"named tuple","value":[{"name":"field1"}]}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        let s = r#"{"kind":"named tuple","value":[{"name":"field1","value":123123}]}"#;
        assert!(serde_json::from_str::<TypedValue>(&s).is_err());
        Ok(())
    }

    #[test]
    fn test_non_human_readable() -> Result<()> {
        // Complicated TypedValue.
        let s = r#"{"kind":"named tuple","value":[{"name":"field1","value":{"kind":"tuple","type":"i32","value":[{"kind":"scalar","type":"u8","value":128},{"kind":"scalar", "type": "b", "value":true}]}},{"name":"field2","value":{"kind":"vector","value":[{"kind":"scalar","type":"i32","value":-123456},{"kind":"scalar","type":"i32","value":123456},{"kind":"scalar","type":"i32","value":-13579},{"kind":"scalar","type":"i32","value":13579}]}}]}"#;
        let tv = serde_json::from_str::<TypedValue>(&s)?;
        let result = bincode::serialize(&tv).unwrap();
        assert_eq!(bincode::deserialize::<TypedValue>(&result).unwrap(), tv);
        Ok(())
    }
}
