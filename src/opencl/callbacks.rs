use std::ffi::c_void;

/// Тип callback-функции для контекста OpenCL
pub type ContextNotifyCallback = Option<
    unsafe extern "C" fn(
        errinfo: *const i8,
        private_info: *const c_void,
        cb: usize,
        user_data: *mut c_void,
    )
>;

/// Пустая callback-функция для контекста
pub unsafe extern "C" fn empty_context_callback(
    _errinfo: *const i8,
    _private_info: *const c_void,
    _cb: usize,
    _user_data: *mut c_void,
) {
    // Ничего не делаем
} 