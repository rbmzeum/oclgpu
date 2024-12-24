//! Вспомогательные функции и макросы для OpenCL

/// Преобразует строку в null-terminated массив байт для C
pub fn to_c_string(s: &str) -> Vec<i8> {
    let mut result: Vec<i8> = s.bytes().map(|b| b as i8).collect();
    result.push(0);
    result
} 