//! Низкоуровневые привязки к OpenCL API

use super::types::*;
use std::ffi::c_void;

#[link(name = "OpenCL")]
unsafe extern "C" {
    pub fn clGetPlatformIDs(
        num_entries: u32,
        platforms: *mut cl_platform_id,
        num_platforms: *mut u32
    ) -> cl_int;

    pub fn clGetPlatformInfo(
        platform: cl_platform_id,
        param_name: cl_platform_info,
        param_value_size: usize,
        param_value: *mut std::ffi::c_void,
        param_value_size_ret: *mut usize
    ) -> cl_int;

    pub fn clGetDeviceIDs(
        platform: cl_platform_id,
        device_type: cl_device_type,
        num_entries: u32,
        devices: *mut cl_device_id,
        num_devices: *mut u32
    ) -> cl_int;

    pub fn clCreateContext(
        properties: *const cl_context_properties,
        num_devices: u32,
        devices: *const cl_device_id,
        pfn_notify: Option<extern "C" fn()>,
        user_data: *mut c_void,
        errcode_ret: *mut cl_int
    ) -> cl_context;

    pub fn clCreateCommandQueue(
        context: cl_context,
        device: cl_device_id,
        properties: cl_command_queue_properties,
        errcode_ret: *mut cl_int
    ) -> cl_command_queue;

    pub fn clCreateProgramWithSource(
        context: cl_context,
        count: u32,
        strings: *const *const i8,
        lengths: *const usize,
        errcode_ret: *mut cl_int
    ) -> cl_program;

    pub fn clBuildProgram(
        program: cl_program,
        num_devices: u32,
        device_list: *const cl_device_id,
        options: *const i8,
        pfn_notify: Option<extern "C" fn()>,
        user_data: *mut c_void
    ) -> cl_int;

    pub fn clCreateBuffer(
        context: cl_context,
        flags: cl_mem_flags,
        size: usize,
        host_ptr: *mut c_void,
        errcode_ret: *mut cl_int
    ) -> cl_mem;

    pub fn clCreateKernel(
        program: cl_program,
        kernel_name: *const i8,
        errcode_ret: *mut cl_int
    ) -> cl_kernel;

    pub fn clSetKernelArg(
        kernel: cl_kernel,
        arg_index: u32,
        arg_size: usize,
        arg_value: *const c_void
    ) -> cl_int;

    pub fn clEnqueueNDRangeKernel(
        command_queue: cl_command_queue,
        kernel: cl_kernel,
        work_dim: u32,
        global_work_offset: *const usize,
        global_work_size: *const usize,
        local_work_size: *const usize,
        num_events_in_wait_list: u32,
        event_wait_list: *const cl_event,
        event: *mut cl_event
    ) -> cl_int;

    pub fn clEnqueueReadBuffer(
        command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_read: bool,
        offset: usize,
        size: usize,
        ptr: *mut c_void,
        num_events_in_wait_list: u32,
        event_wait_list: *const cl_event,
        event: *mut cl_event
    ) -> cl_int;

    pub fn clEnqueueWriteBuffer(
        command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_write: bool,
        offset: usize,
        size: usize,
        ptr: *const c_void,
        num_events_in_wait_list: u32,
        event_wait_list: *const cl_event,
        event: *mut cl_event
    ) -> cl_int;

    pub fn clFinish(command_queue: cl_command_queue) -> cl_int;

    pub fn clReleaseMemObject(memobj: cl_mem) -> cl_int;
    pub fn clReleaseKernel(kernel: cl_kernel) -> cl_int;
    pub fn clReleaseProgram(program: cl_program) -> cl_int;
    pub fn clReleaseCommandQueue(command_queue: cl_command_queue) -> cl_int;
    pub fn clReleaseContext(context: cl_context) -> cl_int;
    pub fn clGetProgramBuildInfo(
        program: cl_program,
        device: cl_device_id,
        param_name: cl_program_build_info,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize
    ) -> cl_int;

    pub fn clGetDeviceInfo(
        device: cl_device_id,
        param_name: cl_device_info,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize
    ) -> cl_int;
} 