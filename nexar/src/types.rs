/// Unique identifier for a node in the cluster.
pub type NodeId = u32;

/// Rank of a participant in a communicator group (0-indexed).
pub type Rank = u32;

/// Data types supported by nexar for collective operations.
///
/// nexar defines its own type enum so it remains
/// a standalone library usable by any Rust project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DataType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    I8 = 4,
    I32 = 5,
    I64 = 6,
    U8 = 7,
    U32 = 8,
    U64 = 9,
}

impl DataType {
    /// Size of one element in bytes.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F64 | DataType::I64 | DataType::U64 => 8,
            DataType::F16 | DataType::BF16 => 2,
            DataType::I8 | DataType::U8 => 1,
        }
    }

    /// Human-readable name.
    pub const fn name(self) -> &'static str {
        match self {
            DataType::F32 => "f32",
            DataType::F64 => "f64",
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::I8 => "i8",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::U8 => "u8",
            DataType::U32 => "u32",
            DataType::U64 => "u64",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Reduction operations for collective communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Element-wise sum across ranks.
    Sum,
    /// Element-wise product across ranks.
    Prod,
    /// Element-wise minimum across ranks.
    Min,
    /// Element-wise maximum across ranks.
    Max,
}

impl std::fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReduceOp::Sum => f.write_str("sum"),
            ReduceOp::Prod => f.write_str("prod"),
            ReduceOp::Min => f.write_str("min"),
            ReduceOp::Max => f.write_str("max"),
        }
    }
}

/// Priority levels for QUIC stream allocation.
///
/// Critical messages (barriers, health) get dedicated streams.
/// Bulk data (tensor transfers) use separate streams to avoid
/// head-of-line blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Priority {
    /// Highest priority: barriers, error signals.
    Critical = 0,
    /// Medium priority: control messages, small tensors.
    Realtime = 1,
    /// Lowest priority: bulk tensor data transfers.
    Bulk = 2,
}

/// Current protocol version.
pub const PROTOCOL_VERSION: u16 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datatype_sizes() {
        assert_eq!(DataType::F32.size_in_bytes(), 4);
        assert_eq!(DataType::F64.size_in_bytes(), 8);
        assert_eq!(DataType::F16.size_in_bytes(), 2);
        assert_eq!(DataType::BF16.size_in_bytes(), 2);
        assert_eq!(DataType::I8.size_in_bytes(), 1);
        assert_eq!(DataType::I32.size_in_bytes(), 4);
        assert_eq!(DataType::I64.size_in_bytes(), 8);
        assert_eq!(DataType::U8.size_in_bytes(), 1);
        assert_eq!(DataType::U32.size_in_bytes(), 4);
        assert_eq!(DataType::U64.size_in_bytes(), 8);
    }

    #[test]
    fn test_datatype_display() {
        assert_eq!(DataType::F32.to_string(), "f32");
        assert_eq!(DataType::BF16.to_string(), "bf16");
        assert_eq!(DataType::I8.to_string(), "i8");
    }

    #[test]
    fn test_datatype_names() {
        let all = [
            DataType::F32,
            DataType::F64,
            DataType::F16,
            DataType::BF16,
            DataType::I8,
            DataType::I32,
            DataType::I64,
            DataType::U8,
            DataType::U32,
            DataType::U64,
        ];
        for dt in all {
            assert!(!dt.name().is_empty());
        }
    }

    #[test]
    fn test_reduce_op_variants() {
        let ops = [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Min, ReduceOp::Max];
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j]);
            }
        }
    }

    #[test]
    fn test_reduce_op_display() {
        assert_eq!(ReduceOp::Sum.to_string(), "sum");
        assert_eq!(ReduceOp::Prod.to_string(), "prod");
        assert_eq!(ReduceOp::Min.to_string(), "min");
        assert_eq!(ReduceOp::Max.to_string(), "max");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical < Priority::Realtime);
        assert!(Priority::Realtime < Priority::Bulk);
    }

    #[test]
    fn test_priority_repr() {
        assert_eq!(Priority::Critical as u8, 0);
        assert_eq!(Priority::Realtime as u8, 1);
        assert_eq!(Priority::Bulk as u8, 2);
    }

    #[test]
    fn test_datatype_repr() {
        assert_eq!(DataType::F32 as u8, 0);
        assert_eq!(DataType::F64 as u8, 1);
        assert_eq!(DataType::U64 as u8, 9);
    }
}
