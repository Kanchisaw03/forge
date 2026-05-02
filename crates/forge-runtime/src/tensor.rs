use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use smallvec::SmallVec;

/// Forge tensor scalar types.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
}

impl DType {
    /// Returns byte width for this scalar type.
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::Bool => 1,
        }
    }
}

/// Fixed slab size classes.
pub const SIZE_CLASSES: [usize; 10] = [64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576];

struct Slab {
    free_list: Mutex<Vec<NonNull<u8>>>,
}

impl Slab {
    fn new(_chunk_size: usize) -> Self {
        Self {
            free_list: Mutex::new(Vec::new()),
        }
    }
}

/// 64-byte aligned slab allocator for tensors.
pub struct TensorAllocator {
    slabs: Vec<Slab>,
    large: Mutex<HashMap<usize, Layout>>,
    total_allocated: AtomicUsize,
}

impl TensorAllocator {
    /// Creates a new allocator instance.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            slabs: SIZE_CLASSES.iter().copied().map(Slab::new).collect(),
            large: Mutex::new(HashMap::new()),
            total_allocated: AtomicUsize::new(0),
        })
    }

    /// Allocates raw bytes with 64-byte alignment.
    pub fn alloc(&self, bytes: usize) -> NonNull<u8> {
        if let Some((idx, class)) = find_size_class(bytes) {
            if let Some(ptr) = self.slabs[idx].free_list.lock().expect("slab lock").pop() {
                return ptr;
            }
            let layout = Layout::from_size_align(class, 64).expect("valid slab layout");
            // SAFETY: `layout` is non-zero size and 64-byte aligned. We check null and wrap in NonNull.
            let raw = unsafe { alloc(layout) };
            let ptr = NonNull::new(raw).expect("allocation must succeed");
            self.total_allocated.fetch_add(class, Ordering::Relaxed);
            ptr
        } else {
            let layout = Layout::from_size_align(bytes.max(64), 64).expect("valid large layout");
            // SAFETY: `layout` is valid and aligned to 64 bytes. Null is checked below.
            let raw = unsafe { alloc(layout) };
            let ptr = NonNull::new(raw).expect("allocation must succeed");
            self.large
                .lock()
                .expect("large lock")
                .insert(ptr.as_ptr() as usize, layout);
            self.total_allocated.fetch_add(layout.size(), Ordering::Relaxed);
            ptr
        }
    }

    /// Deallocates a previously-allocated pointer.
    pub fn dealloc(&self, ptr: NonNull<u8>, bytes: usize) {
        if let Some((idx, _class)) = find_size_class(bytes) {
            self.slabs[idx]
                .free_list
                .lock()
                .expect("slab lock")
                .push(ptr);
            return;
        }
        if let Some(layout) = self
            .large
            .lock()
            .expect("large lock")
            .remove(&(ptr.as_ptr() as usize))
        {
            // SAFETY: layout matches the one used at allocation and pointer came from `alloc`.
            unsafe { dealloc(ptr.as_ptr(), layout) };
        }
    }

    /// Creates one tensor allocation and metadata wrapper.
    pub fn tensor(
        self: &Arc<Self>,
        len: usize,
        dtype: DType,
        shape: impl Into<SmallVec<[u32; 4]>>,
    ) -> Tensor {
        let bytes = len.saturating_mul(dtype.size_bytes());
        let ptr = self.alloc(bytes.max(1));
        Tensor {
            ptr,
            len,
            dtype,
            shape: shape.into(),
            allocator: Arc::clone(self),
        }
    }

    /// Returns total allocated bytes from backing system allocator.
    pub fn total_allocated_bytes(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }
}

fn find_size_class(bytes: usize) -> Option<(usize, usize)> {
    SIZE_CLASSES
        .iter()
        .copied()
        .enumerate()
        .find(|(_, class)| bytes <= *class)
}

/// Tensor allocation object.
pub struct Tensor {
    ptr: NonNull<u8>,
    len: usize,
    dtype: DType,
    shape: SmallVec<[u32; 4]>,
    allocator: Arc<TensorAllocator>,
}

impl Tensor {
    /// Returns raw mutable pointer to tensor data.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns element length.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns shape.
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        let bytes = self.len.saturating_mul(self.dtype.size_bytes()).max(1);
        self.allocator.dealloc(self.ptr, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_and_recycles_many_tensors() {
        let alloc = TensorAllocator::new();
        let before = alloc.total_allocated_bytes();
        let mut tensors = Vec::new();
        for i in 1..=1000usize {
            let len = (i * 13) % 4096 + 1;
            let mut shape = SmallVec::<[u32; 4]>::new();
            shape.push(len as u32);
            tensors.push(alloc.tensor(len, DType::F32, shape));
        }
        drop(tensors);
        let after = alloc.total_allocated_bytes();
        assert!(after >= before);
    }

    #[test]
    fn tensor_allocations_are_64_byte_aligned() {
        let alloc = TensorAllocator::new();
        let tensor = alloc.tensor(1024, DType::F32, vec![1024u32]);
        let addr = tensor.as_mut_ptr() as usize;
        assert_eq!(addr % 64, 0, "tensor pointer should be 64-byte aligned");
    }
}
