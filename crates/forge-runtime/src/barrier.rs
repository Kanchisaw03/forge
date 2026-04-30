use std::hint::spin_loop;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Sense-reversing barrier for block synchronization.
pub struct SenseBarrier {
    count: AtomicU32,
    total: u32,
    sense: AtomicBool,
}

impl SenseBarrier {
    /// Creates a barrier that waits for `total` participants.
    pub fn new(total: u32) -> Self {
        Self {
            count: AtomicU32::new(total),
            total,
            sense: AtomicBool::new(false),
        }
    }

    /// Waits until all participants arrive at this barrier epoch.
    pub fn wait(&self) {
        let local_sense = !self.sense.load(Ordering::Relaxed);
        let prev = self.count.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.count.store(self.total, Ordering::Release);
            self.sense.store(local_sense, Ordering::Release);
            return;
        }

        let mut spins = 0u32;
        while self.sense.load(Ordering::Acquire) != local_sense {
            if spins < 10 {
                spin_loop();
            } else {
                std::thread::yield_now();
            }
            spins = spins.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::thread;

    use super::*;

    #[test]
    fn all_threads_wait_for_each_other() {
        let barrier = Arc::new(SenseBarrier::new(16));
        let arrived = Arc::new(AtomicU32::new(0));
        let mut handles = Vec::new();
        for _ in 0..16 {
            let b = Arc::clone(&barrier);
            let a = Arc::clone(&arrived);
            handles.push(thread::spawn(move || {
                a.fetch_add(1, Ordering::SeqCst);
                b.wait();
                assert_eq!(a.load(Ordering::SeqCst), 16);
            }));
        }
        for h in handles {
            h.join().expect("thread join must succeed");
        }
    }
}
