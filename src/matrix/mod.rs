//! Модуль для работы с матрицами
//! 
//! Предоставляет:
//! - Типы матриц
//! - Операции над матрицами
//! - GPU-ускоренные реализации

mod types;
pub mod operations;
pub mod kernels;

pub use types::MatrixType;
pub use operations::{cpu_matrix_multiply, compare_results, initialize_matrices};
pub use kernels::MATRIX_MULTIPLY_KERNEL; 