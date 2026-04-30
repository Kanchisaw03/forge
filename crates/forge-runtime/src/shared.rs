/// Per-block shared memory simulation.
///
/// In the GPU model, each block gets a fixed-size scratchpad in on-chip memory.
/// We model this as a 32-byte aligned heap allocation so that AVX2 loads
/// (`_mm256_load_ps`) work correctly.
pub struct SharedMemory {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    size: usize,
}

// SAFETY: SharedMemory owns its allocation exclusively and does not use
// interior mutability or thread-local state.
unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Allocates `size` bytes of 32-byte aligned shared memory for one block.
    pub fn new(size: usize) -> Self {
        let size = size.max(1);
        // Round up to 32-byte boundary so the full allocation is aligned.
        let alloc_size = (size + 31) & !31;
        let layout = std::alloc::Layout::from_size_align(alloc_size, 32)
            .expect("SharedMemory layout must be valid");
        // SAFETY: layout has non-zero size and valid alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "SharedMemory allocation failed");
        Self { ptr, layout, size }
    }

    /// Returns a raw mutable pointer to the underlying buffer.
    ///
    /// # Safety
    /// The caller must ensure accesses stay within `[0, size)` bytes and
    /// that access patterns do not create data races when multiple simulated
    /// threads run on the same block (they must use barrier synchronisation).
    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Returns the usable size in bytes.
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns a typed mutable slice view into the shared buffer.
    ///
    /// # Safety
    /// Same aliasing rules as `as_mut_ptr`.
    pub unsafe fn as_slice_mut<T>(&mut self) -> &mut [T] {
        let ptr = self.as_mut_ptr() as *mut T;
        let len = self.size / std::mem::size_of::<T>();
        std::slice::from_raw_parts_mut(ptr, len)
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this exact layout in `new`.
        unsafe {
            std::alloc::dealloc(self.ptr, self.layout);
        }
    }
}

/// Type-erased kernel argument buffer.
///
/// The design specifies that arguments are packed into a byte buffer so the
/// launcher can forward them to the kernel without knowing the concrete types
/// at compile time. This mirrors how CUDA/HIP drivers serialize kernel args.
pub struct KernelArgs {
    data: Vec<u8>,
    offsets: Vec<usize>,
}

impl KernelArgs {
    /// Creates an empty argument buffer.
    pub fn new() -> Self {
        Self { data: Vec::new(), offsets: Vec::new() }
    }

    /// Returns number of packed arguments.
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Returns true when no arguments have been packed.
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Returns a raw pointer to argument `idx`.
    ///
    /// # Safety
    /// Caller must cast to the correct type and must not retain the pointer
    /// past the lifetime of this `KernelArgs`.
    pub unsafe fn get_raw(&self, idx: usize) -> *const u8 {
        self.data.as_ptr().add(self.offsets[idx])
    }
}

impl Default for KernelArgs {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for `KernelArgs`.
///
/// ```rust,ignore
/// let args = ArgBuilder::new()
///     .push(a_ptr as *const f32)
///     .push(b_ptr as *const f32)
///     .push(n as u32)
///     .build();
/// ```
pub struct ArgBuilder {
    inner: KernelArgs,
}

impl ArgBuilder {
    /// Creates a builder with an empty argument list.
    pub fn new() -> Self {
        Self { inner: KernelArgs::new() }
    }

    /// Packs one argument by value.
    pub fn push<T: Copy>(mut self, value: T) -> Self {
        let offset = self.inner.data.len();
        self.inner.offsets.push(offset);
        // SAFETY: We write exactly `size_of::<T>()` bytes from a valid reference.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &value as *const T as *const u8,
                std::mem::size_of::<T>(),
            )
        };
        self.inner.data.extend_from_slice(bytes);
        self
    }

    /// Consumes the builder and returns the packed `KernelArgs`.
    pub fn build(self) -> KernelArgs {
        self.inner
    }
}

impl Default for ArgBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_memory_is_32_byte_aligned() {
        let mut shm = SharedMemory::new(1024);
        let ptr = unsafe { shm.as_mut_ptr() };
        assert_eq!(ptr as usize % 32, 0, "shared memory must be 32-byte aligned");
    }

    #[test]
    fn arg_builder_round_trips_values() {
        let args = ArgBuilder::new()
            .push(42u32)
            .push(3.14f32)
            .push(100i32)
            .build();
        assert_eq!(args.len(), 3);
        unsafe {
            let v0 = *(args.get_raw(0) as *const u32);
            let v1 = *(args.get_raw(1) as *const f32);
            let v2 = *(args.get_raw(2) as *const i32);
            assert_eq!(v0, 42u32);
            assert!((v1 - 3.14f32).abs() < 1e-6);
            assert_eq!(v2, 100i32);
        }
    }
}
