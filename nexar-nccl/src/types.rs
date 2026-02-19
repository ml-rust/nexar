use cudarc::nccl::sys;
use nexar::types::{DataType, ReduceOp};

/// Convert nexar DataType to NCCL data type.
pub fn to_nccl_dtype(dt: DataType) -> sys::ncclDataType_t {
    match dt {
        DataType::F32 => sys::ncclDataType_t::ncclFloat32,
        DataType::F64 => sys::ncclDataType_t::ncclFloat64,
        DataType::F16 => sys::ncclDataType_t::ncclFloat16,
        DataType::BF16 => sys::ncclDataType_t::ncclBfloat16,
        DataType::I8 => sys::ncclDataType_t::ncclInt8,
        DataType::I32 => sys::ncclDataType_t::ncclInt32,
        DataType::I64 => sys::ncclDataType_t::ncclInt64,
        DataType::U8 => sys::ncclDataType_t::ncclUint8,
        DataType::U32 => sys::ncclDataType_t::ncclUint32,
        DataType::U64 => sys::ncclDataType_t::ncclUint64,
    }
}

/// Convert nexar ReduceOp to NCCL reduce operation.
pub fn to_nccl_op(op: ReduceOp) -> sys::ncclRedOp_t {
    match op {
        ReduceOp::Sum => sys::ncclRedOp_t::ncclSum,
        ReduceOp::Prod => sys::ncclRedOp_t::ncclProd,
        ReduceOp::Min => sys::ncclRedOp_t::ncclMin,
        ReduceOp::Max => sys::ncclRedOp_t::ncclMax,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(
            to_nccl_dtype(DataType::F32),
            sys::ncclDataType_t::ncclFloat32
        );
        assert_eq!(
            to_nccl_dtype(DataType::BF16),
            sys::ncclDataType_t::ncclBfloat16
        );
        assert_eq!(to_nccl_dtype(DataType::U8), sys::ncclDataType_t::ncclUint8);
    }

    #[test]
    fn test_op_mapping() {
        assert_eq!(to_nccl_op(ReduceOp::Sum), sys::ncclRedOp_t::ncclSum);
        assert_eq!(to_nccl_op(ReduceOp::Max), sys::ncclRedOp_t::ncclMax);
    }
}
