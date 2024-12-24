//! OpenCL типы данных

#[allow(non_camel_case_types)]
pub type cl_platform_id = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type cl_device_id = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type cl_context = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type cl_command_queue = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type cl_program = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type cl_kernel = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type cl_mem = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type cl_event = *mut std::ffi::c_void;
#[allow(non_camel_case_types)] 
pub type cl_platform_info = u32;
#[allow(non_camel_case_types)]
pub type cl_device_type = u64;
#[allow(non_camel_case_types)]
pub type cl_int = i32;
#[allow(non_camel_case_types)]
pub type cl_context_properties = isize;
#[allow(non_camel_case_types)]
pub type cl_command_queue_properties = u64;
#[allow(non_camel_case_types)]
pub type cl_mem_flags = u64;
#[allow(non_camel_case_types)]
pub type cl_program_build_info = u32;

// Константы OpenCL
pub const CL_DEVICE_TYPE_CPU: cl_device_type = 1 << 1;
pub const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;
pub const CL_MEM_READ_ONLY: cl_mem_flags = 1 << 0;
pub const CL_MEM_WRITE_ONLY: cl_mem_flags = 1 << 1;
pub const CL_MEM_COPY_HOST_PTR: cl_mem_flags = 1 << 5;
pub const CL_MEM_ALLOC_HOST_PTR: cl_mem_flags = 1 << 4;
pub const CL_PROGRAM_BUILD_LOG: cl_program_build_info = 0x1183; 