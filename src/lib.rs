//! OpenCL-accelerated neural network operations

pub mod matrix;
pub mod neural;
pub mod opencl;
pub mod utils;

// Реэкспортируем макросы на уровень крейта
#[macro_use]
mod macros {
    /// Макрос для обработки ошибок OpenCL (коды возврата)
    #[macro_export]
    macro_rules! cl_check {
        ($func:expr) => {{
            let status = $func;
            if status != 0 {
                Err(anyhow::anyhow!("OpenCL error code: {}", status))
            } else {
                Ok(())
            }
        }};
    }

    /// Макрос для обработки указателей OpenCL
    #[macro_export]
    macro_rules! cl_create {
        ($func:expr) => {{
            let ptr = $func;
            if ptr.is_null() {
                Err(anyhow::anyhow!("OpenCL error: null pointer returned"))
            } else {
                Ok(ptr)
            }
        }};
    }
}

// Реэкспорт основных типов для удобства
pub use matrix::MatrixType;
pub use opencl::types::*; 