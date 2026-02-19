//! Integration tests for the hierarchical communicator.
//!
//! These tests require CUDA GPUs and NCCL to be available.
//! Run with: cargo test -p nexar-nccl
//!
//! On systems without CUDA, these tests will fail at NCCL initialization.
//! Gate execution with: `NEXAR_NCCL_TEST=1 cargo test -p nexar-nccl`

use nexar::types::{DataType, ReduceOp};
use nexar_nccl::topology::NodeTopology;

#[test]
fn test_type_mappings() {
    use cudarc::nccl::sys;
    use nexar_nccl::{to_nccl_dtype, to_nccl_op};

    assert_eq!(
        to_nccl_dtype(DataType::F32),
        sys::ncclDataType_t::ncclFloat32
    );
    assert_eq!(
        to_nccl_dtype(DataType::F64),
        sys::ncclDataType_t::ncclFloat64
    );
    assert_eq!(
        to_nccl_dtype(DataType::F16),
        sys::ncclDataType_t::ncclFloat16
    );
    assert_eq!(
        to_nccl_dtype(DataType::BF16),
        sys::ncclDataType_t::ncclBfloat16
    );
    assert_eq!(to_nccl_dtype(DataType::I8), sys::ncclDataType_t::ncclInt8);
    assert_eq!(to_nccl_dtype(DataType::I32), sys::ncclDataType_t::ncclInt32);
    assert_eq!(to_nccl_dtype(DataType::I64), sys::ncclDataType_t::ncclInt64);
    assert_eq!(to_nccl_dtype(DataType::U8), sys::ncclDataType_t::ncclUint8);
    assert_eq!(
        to_nccl_dtype(DataType::U32),
        sys::ncclDataType_t::ncclUint32
    );
    assert_eq!(
        to_nccl_dtype(DataType::U64),
        sys::ncclDataType_t::ncclUint64
    );

    assert_eq!(to_nccl_op(ReduceOp::Sum), sys::ncclRedOp_t::ncclSum);
    assert_eq!(to_nccl_op(ReduceOp::Prod), sys::ncclRedOp_t::ncclProd);
    assert_eq!(to_nccl_op(ReduceOp::Min), sys::ncclRedOp_t::ncclMin);
    assert_eq!(to_nccl_op(ReduceOp::Max), sys::ncclRedOp_t::ncclMax);
}

#[test]
fn test_nccl_id_serialization() {
    use nexar_nccl::group::{id_from_bytes, id_to_bytes};

    let mut internal = [0i8; 128];
    for (i, val) in internal.iter_mut().enumerate() {
        *val = (i % 127) as i8;
    }
    let id = cudarc::nccl::safe::Id::uninit(internal);

    let bytes = id_to_bytes(&id);
    assert_eq!(bytes.len(), 128);

    let recovered = id_from_bytes(&bytes);
    let recovered_bytes = id_to_bytes(&recovered);
    assert_eq!(bytes, recovered_bytes);
}

#[test]
fn test_node_topology_construction() {
    let topo = NodeTopology {
        hostname: "node0".into(),
        local_ranks: vec![0, 1, 2, 3],
        local_rank_idx: 0,
        lead_rank: 0,
        inter_node_leads: vec![0, 4],
        node_idx: 0,
        num_nodes: 2,
    };

    assert!(topo.is_lead());
    assert!(!topo.is_single_node());
    assert_eq!(topo.local_world_size(), 4);
    assert_eq!(topo.num_nodes, 2);
}

#[test]
fn test_node_topology_non_lead() {
    let topo = NodeTopology {
        hostname: "node1".into(),
        local_ranks: vec![4, 5, 6, 7],
        local_rank_idx: 2,
        lead_rank: 4,
        inter_node_leads: vec![0, 4],
        node_idx: 1,
        num_nodes: 2,
    };

    assert!(!topo.is_lead());
    assert_eq!(topo.local_world_size(), 4);
}

#[test]
fn test_single_node_topology() {
    let topo = NodeTopology {
        hostname: "node0".into(),
        local_ranks: vec![0, 1],
        local_rank_idx: 0,
        lead_rank: 0,
        inter_node_leads: vec![0],
        node_idx: 0,
        num_nodes: 1,
    };

    assert!(topo.is_single_node());
    assert!(topo.is_lead());
}

#[test]
fn test_error_types() {
    use nexar_nccl::NcclCommError;

    let e = NcclCommError::Topology {
        reason: "test".into(),
    };
    assert!(e.to_string().contains("test"));

    let e = NcclCommError::InvalidRank {
        rank: 5,
        world_size: 4,
    };
    assert!(e.to_string().contains("5"));

    let e = NcclCommError::NotLeadRank { rank: 3 };
    assert!(e.to_string().contains("3"));

    let e = NcclCommError::Bootstrap {
        reason: "failed".into(),
    };
    assert!(e.to_string().contains("failed"));
}
