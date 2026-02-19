mod adapter;
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;

pub use adapter::DeviceAdapter;
pub use cpu::CpuAdapter;
#[cfg(feature = "cuda")]
pub use cuda::CudaAdapter;
