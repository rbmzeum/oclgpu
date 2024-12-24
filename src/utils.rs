//! Вспомогательные функции и утилиты

use std::time::Instant;

/// Измеряет время выполнения функции
pub fn measure_time<F, T>(f: F) -> (T, std::time::Duration) 
where 
    F: FnOnce() -> T
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
} 