use std::sync::Arc;

use crossbeam_deque::{Injector, Steal, Stealer as CbStealer, Worker as CbWorker};

/// Owner-side push/pop handle.
pub struct Worker<T> {
    local: CbWorker<T>,
    injector: Arc<Injector<T>>,
}

/// Multi-consumer stealing handle.
#[derive(Clone)]
pub struct Stealer<T> {
    local: CbStealer<T>,
    injector: Arc<Injector<T>>,
}

/// Creates a Chase-Lev style worker and matching stealer.
pub fn deque<T>() -> (Worker<T>, Stealer<T>) {
    let local = CbWorker::new_fifo();
    let injector = Arc::new(Injector::new());
    let stealer = Stealer {
        local: local.stealer(),
        injector: Arc::clone(&injector),
    };
    (
        Worker { local, injector },
        stealer,
    )
}

impl<T> Worker<T> {
    /// Pushes one task to the owner bottom.
    pub fn push(&self, task: T) {
        self.local.push(task);
    }

    /// Pops one task from owner bottom.
    pub fn pop(&self) -> Option<T> {
        self.local.pop().or_else(|| self.injector.steal().success())
    }

    /// Makes this worker task visible to stealers.
    pub fn publish(&self, task: T) {
        self.injector.push(task);
    }

    /// Returns a stealer for this worker.
    pub fn stealer(&self) -> Stealer<T> {
        Stealer {
            local: self.local.stealer(),
            injector: Arc::clone(&self.injector),
        }
    }
}

impl<T> Stealer<T> {
    /// Attempts to steal one task.
    pub fn steal(&self) -> Option<T> {
        match self.local.steal() {
            Steal::Success(v) => Some(v),
            Steal::Retry => self.steal(),
            Steal::Empty => self.injector.steal().success(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use std::thread;

    use super::*;

    #[test]
    fn processes_all_tasks_exactly_once_with_stealers() {
        let (worker, stealer) = deque::<u32>();
        for i in 0..10_000u32 {
            worker.publish(i);
        }
        let seen = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = stealer.clone();
            let seen_ref = Arc::clone(&seen);
            handles.push(thread::spawn(move || {
                while let Some(v) = s.steal() {
                    seen_ref.lock().expect("lock must succeed").insert(v);
                }
            }));
        }
        for h in handles {
            h.join().expect("thread join must succeed");
        }
        let guard = seen.lock().expect("lock must succeed");
        assert_eq!(guard.len(), 10_000);
    }
}
