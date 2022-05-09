use crate::errors::Result;
use serde::{Deserialize, Serialize};
//DATA_VERSION represents the current version of serializable data types, i.e., Value and Context. We only bump this const when we make backwards-incompatible changes of Value and Context.
pub(crate) const DATA_VERSION: u64 = 1;

#[derive(Serialize, Deserialize)]
pub(crate) struct VersionedData {
    version: u64,
    data: String,
}

impl VersionedData {
    pub(crate) fn create_versioned_data(ver: u64, serialized_data: String) -> Result<Self> {
        Ok(VersionedData {
            version: ver,
            data: serialized_data,
        })
    }

    pub(crate) fn check_version(&self, version_req: u64) -> bool {
        self.version == version_req
    }

    pub(crate) fn get_data_string(&self) -> &str {
        &self.data
    }
}
