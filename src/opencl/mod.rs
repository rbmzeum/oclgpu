//! Модуль для работы с OpenCL

pub mod bindings;
pub mod types;
pub mod utils;

pub use utils::*;
pub use bindings::*;
pub use types::*;
pub use crate::{cl_check, cl_create}; 