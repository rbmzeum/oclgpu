//! Модуль для работы с OpenCL
//! 
//! Содержит низкоуровневые привязки и безопасные обертки для OpenCL

pub mod bindings;
pub mod types;
pub mod utils;

pub use utils::*;
// Реэкспортируем макросы из корня крейта
pub use crate::{cl_check, cl_create}; 