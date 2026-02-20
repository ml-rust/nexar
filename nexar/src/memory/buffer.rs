//! Typed buffer wrappers that encode memory space in the type system.
//!
//! These are zero-cost wrappers around raw `u64` pointers. The type parameter
//! prevents accidentally passing a host pointer where a device pointer is
//! expected (and vice versa).
//!
//! The raw `u64` API remains for backward compatibility and FFI use cases.
//! These wrappers are opt-in for users who want compile-time memory space safety.

use std::marker::PhantomData;

// ── Sealed trait pattern ─────────────────────────────────────────────

mod private {
    pub trait Sealed {}
}

/// Marker trait for memory spaces (host vs device).
pub trait MemorySpace: private::Sealed {}

/// Host (CPU) memory.
pub enum Host {}
impl private::Sealed for Host {}
impl MemorySpace for Host {}

/// Device (GPU) memory.
pub enum Device {}
impl private::Sealed for Device {}
impl MemorySpace for Device {}

// ── BufferPtr ────────────────────────────────────────────────────────

/// A typed pointer to memory in a specific memory space.
///
/// Zero-cost wrapper around a raw `u64` pointer. The type parameter `S`
/// prevents accidentally passing a host pointer where a device pointer
/// is expected.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferPtr<S: MemorySpace> {
    ptr: u64,
    _space: PhantomData<S>,
}

impl<S: MemorySpace> BufferPtr<S> {
    /// Wrap a raw `u64` pointer.
    ///
    /// # Safety
    /// The pointer must actually point to memory in the space `S`.
    pub unsafe fn new(ptr: u64) -> Self {
        Self {
            ptr,
            _space: PhantomData,
        }
    }

    /// Get the raw `u64` pointer.
    pub fn as_u64(&self) -> u64 {
        self.ptr
    }
}

impl<S: MemorySpace> std::fmt::Display for BufferPtr<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BufferPtr(0x{:x})", self.ptr)
    }
}

// ── BufferRef ────────────────────────────────────────────────────────

/// A typed, sized buffer reference in a specific memory space.
///
/// Pairs a [`BufferPtr`] with a byte length, providing both type-level
/// memory space safety and runtime size information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferRef<S: MemorySpace> {
    ptr: BufferPtr<S>,
    len_bytes: usize,
}

impl<S: MemorySpace> BufferRef<S> {
    /// Create a new buffer reference.
    ///
    /// # Safety
    /// `ptr` must point to at least `len_bytes` of valid memory in space `S`.
    pub unsafe fn new(ptr: u64, len_bytes: usize) -> Self {
        Self {
            ptr: unsafe { BufferPtr::new(ptr) },
            len_bytes,
        }
    }

    /// Get a reference to the typed pointer.
    pub fn ptr(&self) -> &BufferPtr<S> {
        &self.ptr
    }

    /// Size of the buffer in bytes.
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    /// Returns true if the buffer has zero length.
    pub fn is_empty(&self) -> bool {
        self.len_bytes == 0
    }

    /// Get the raw `u64` pointer.
    pub fn as_u64(&self) -> u64 {
        self.ptr.as_u64()
    }
}

impl<S: MemorySpace> std::fmt::Display for BufferRef<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BufferRef(0x{:x}, {}B)",
            self.ptr.as_u64(),
            self.len_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_ptr_host() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let ptr = unsafe { BufferPtr::<Host>::new(data.as_ptr() as u64) };
        assert_eq!(ptr.as_u64(), data.as_ptr() as u64);
    }

    #[test]
    fn test_buffer_ref_size() {
        let data: Vec<u8> = vec![0; 1024];
        let buf = unsafe { BufferRef::<Host>::new(data.as_ptr() as u64, 1024) };
        assert_eq!(buf.len_bytes(), 1024);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_ref_empty() {
        let buf = unsafe { BufferRef::<Host>::new(0, 0) };
        assert!(buf.is_empty());
    }

    #[test]
    fn test_display() {
        let ptr = unsafe { BufferPtr::<Device>::new(0xDEAD) };
        assert!(ptr.to_string().contains("0xdead"));

        let buf = unsafe { BufferRef::<Host>::new(0xFF, 256) };
        let s = buf.to_string();
        assert!(s.contains("0xff"));
        assert!(s.contains("256B"));
    }

    #[test]
    fn test_type_safety_compiles() {
        // This test verifies that Host and Device are distinct types.
        // A function accepting BufferRef<Host> won't accept BufferRef<Device>.
        fn _takes_host(_buf: &BufferRef<Host>) {}
        fn _takes_device(_buf: &BufferRef<Device>) {}

        let host_buf = unsafe { BufferRef::<Host>::new(0x1000, 64) };
        let device_buf = unsafe { BufferRef::<Device>::new(0x2000, 64) };
        _takes_host(&host_buf);
        _takes_device(&device_buf);
    }
}
