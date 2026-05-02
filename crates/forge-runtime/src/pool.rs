use std::sync::{Arc, Once};

use crossbeam_deque::{Injector, Steal, Stealer as CbStealer, Worker as CbWorker};

static INIT_RAYON_GLOBAL_POOL: Once = Once::new();

/// Returns the physical CPU core count used by Forge for compute-bound kernels.
pub fn physical_cpu_count() -> usize {
    num_cpus::get_physical().max(1)
}

/// Returns the Rayon thread count target for compute-bound kernels.
pub fn optimal_thread_count() -> usize {
    if let Some(override_threads) = std::env::var_os("FORGE_RAYON_THREADS") {
        if let Some(parsed) = override_threads.to_str().and_then(|v| v.parse::<usize>().ok()) {
            return parsed.clamp(1, 64);
        }
    }

    let physical = num_cpus::get_physical();
    let logical = num_cpus::get();
    let threads = if physical > 0 {
        physical
    } else {
        (logical / 2).max(1)
    };
    threads.clamp(1, 64)
}

/// Initializes Rayon global pool once, pinned to physical cores.
///
/// Returns the currently active Rayon global thread count.
pub fn init_rayon_global_pool() -> usize {
    INIT_RAYON_GLOBAL_POOL.call_once(|| {
        let num_threads = optimal_thread_count();
        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("forge-worker-{i}"))
            .build_global();

        match result {
            Ok(()) => eprintln!(
                "[forge-runtime] initialized Rayon global pool with {num_threads} threads"
            ),
            Err(err) => eprintln!(
                "[forge-runtime] Rayon global pool already initialized elsewhere: {err}"
            ),
        }
    });

    rayon::current_num_threads()
}

/// Prints current thread diagnostics for benchmark and profiling paths.
pub fn print_thread_diagnostics() {
    println!("Rayon threads:   {}", rayon::current_num_threads());
    println!("Physical cores:  {}", num_cpus::get_physical());
    println!("Logical cores:   {}", num_cpus::get());
}

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

    #[test]
    fn rayon_global_pool_reaches_physical_cores() {
        let configured = init_rayon_global_pool();
        let expected = physical_cpu_count();
        let seen_threads = Arc::new(Mutex::new(HashSet::new()));

        rayon::scope(|scope| {
            for _ in 0..4_096 {
                let seen_threads = Arc::clone(&seen_threads);
                scope.spawn(move |_| {
                    if let Some(idx) = rayon::current_thread_index() {
                        seen_threads.lock().expect("lock must succeed").insert(idx);
                    }
                });
            }
        });

        let unique = seen_threads.lock().expect("lock must succeed").len();
        assert!(
            configured >= 1,
            "rayon should report at least one thread, got {configured}"
        );
        assert!(
            unique + 1 >= expected,
            "expected about {expected} worker threads, observed {unique}"
        );
    }
}
