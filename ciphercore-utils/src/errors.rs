use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Serialize, PartialEq, Eq, Clone, Deserialize)]
#[repr(C)]
pub enum CiphercoreErrorKind {
    RuntimeError,
}

mod custom_date_time_format {

    use chrono::{DateTime, NaiveDateTime, Utc};
    use serde::{self, Deserialize, Deserializer, Serializer};
    const FORMAT: &str = "%Y-%m-%d %H:%M:%S";

    pub fn serialize<S>(date: &DateTime<Utc>, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", date.format(FORMAT));
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        NaiveDateTime::parse_from_str(s.as_str(), FORMAT)
            .map_err(serde::de::Error::custom)
            .map(|x| {
                let now = Utc::now();
                let date: DateTime<Utc> = DateTime::from_utc(x, *now.offset());
                date
            })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CiphercoreErrorBody {
    pub kind: CiphercoreErrorKind,
    pub message: String,
    pub module: String,
    pub file: String,
    pub line: u32,
    pub column: u32,
    #[serde(with = "custom_date_time_format")]
    pub utc_date_time: DateTime<Utc>,
    #[cfg(feature = "nightly-features")]
    pub backtrace: String,
}

impl fmt::Display for CiphercoreErrorBody {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::to_string_pretty(&self).unwrap())
    }
}

pub trait ErrorWithBody {
    fn get_body(&self) -> CiphercoreErrorBody;
}

#[doc(hidden)]
#[macro_export]
macro_rules! runtime_error_body {
    ($($x: expr),*) => {
        $crate::errors::CiphercoreErrorBody {
            kind: $crate::errors::CiphercoreErrorKind::RuntimeError,
            message: format!($($x,)*),
            module: module_path!().to_owned(),
            file: file!().to_owned(),
            line: line!(),
            column: column!(),
            utc_date_time: chrono::Utc::now(),
            #[cfg(feature = "nightly-features")]
            backtrace: std::backtrace::Backtrace::force_capture().to_string(),
        }
    };
}

impl Clone for CiphercoreErrorBody {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            message: self.message.clone(),
            module: self.module.clone(),
            file: self.file.clone(),
            line: self.line,
            column: self.column,
            utc_date_time: self.utc_date_time,
            #[cfg(feature = "nightly-features")]
            backtrace: self.backtrace.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macros() {
        let e = runtime_error_body!("Test {}", 31);
        assert_eq!(e.kind, CiphercoreErrorKind::RuntimeError);
        assert_eq!(e.message, "Test 31");
    }
}
