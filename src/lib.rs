//! OpenCL-accelerated neural network operations

pub mod neural;
pub mod opencl;
pub mod matrix;
pub mod utils;

// Реэкспортируем макросы на уровень крейта
#[macro_use]
mod macros {
    /// Макрос для обработки ошибок OpenCL (коды возврата)
    #[macro_export]
    macro_rules! cl_check {
        ($expr:expr) => {
            unsafe {
                let code = $expr;
                if code != 0 {
                    return Err(anyhow::anyhow!("OpenCL error code: {}", code));
                }
                Ok(()) as anyhow::Result<()>
            }
        };
    }

    /// Макрос для обработки указателей OpenCL
    #[macro_export]
    macro_rules! cl_create {
        // Специальный случай для clCreateContext
        (clCreateContext($props:expr, $num:expr, $devs:expr, $cb:expr, $data:expr, $err:expr)) => {{
            let callback: Option<unsafe extern "C" fn(*const i8, *const std::ffi::c_void, usize, *mut std::ffi::c_void)> = None;
            let obj = unsafe { 
                crate::opencl::bindings::clCreateContext(
                    $props, 
                    $num, 
                    $devs, 
                    callback,
                    std::ptr::null_mut(), 
                    $err
                ) 
            };
            if obj.is_null() {
                return Err(anyhow::anyhow!("Failed to create OpenCL context"));
            }
            Ok(obj) as anyhow::Result<crate::opencl::types::cl_context>
        }};
        // Общий случай для других функций
        ($func:ident($($arg:expr),*)) => {{
            let obj = unsafe { $func($($arg),*) };
            if obj.is_null() {
                return Err(anyhow::anyhow!(concat!("Failed to create OpenCL object: ", stringify!($func))));
            }
            Ok(obj) as anyhow::Result<_>
        }};
    }
}

// Реэкспорт основных типов для удобства
pub use matrix::MatrixType;
pub use opencl::types::*;