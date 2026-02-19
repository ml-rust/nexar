use cudarc::nccl::result::NcclError;
use nexar::Rank;

pub type Result<T> = std::result::Result<T, NcclCommError>;

#[derive(Debug, thiserror::Error)]
pub enum NcclCommError {
    #[error("NCCL error: {0:?}")]
    Nccl(NcclError),

    #[error("nexar error: {0}")]
    Nexar(#[from] nexar::NexarError),

    #[error("CUDA driver error: {0}")]
    CudaDriver(#[from] cudarc::driver::result::DriverError),

    #[error("topology error: {reason}")]
    Topology { reason: String },

    #[error("invalid rank {rank}: world size is {world_size}")]
    InvalidRank { rank: Rank, world_size: u32 },

    #[error("not a lead rank: rank {rank} is not the lead for its node")]
    NotLeadRank { rank: Rank },

    #[error("NCCL bootstrap failed: {reason}")]
    Bootstrap { reason: String },
}

impl From<NcclError> for NcclCommError {
    fn from(e: NcclError) -> Self {
        NcclCommError::Nccl(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = NcclCommError::Topology {
            reason: "hostname mismatch".into(),
        };
        assert!(e.to_string().contains("hostname mismatch"));
    }

    #[test]
    fn test_nexar_error_conversion() {
        let nexar_err = nexar::NexarError::Cancelled;
        let e: NcclCommError = nexar_err.into();
        assert!(e.to_string().contains("cancelled"));
    }
}
