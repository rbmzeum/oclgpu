//! OpenCL-ускоренная библиотека для матричных вычислений и нейронных сетей
//! 
//! Эта библиотека предоставляет высокопроизводительные реализации:
//! - Матричных операций
//! - (Планируется) Нейросетевых вычислений
//! 
//! Использует OpenCL для ускорения на GPU/CPU.

pub mod opencl;
pub mod matrix;
pub mod neural;
pub mod utils;

// Реэкспорт основных типов для удобства
pub use matrix::MatrixType;
pub use opencl::types::*; 