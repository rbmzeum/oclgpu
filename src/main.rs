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
        pfn_notify: *mut std::ffi::c_void,
        user_data: *mut std::ffi::c_void,
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
        pfn_notify: *mut std::ffi::c_void,
        user_data: *mut std::ffi::c_void
    ) -> cl_int;

    pub fn clCreateBuffer(
        context: cl_context,
        flags: cl_mem_flags,
        size: usize,
        host_ptr: *mut std::ffi::c_void,
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
        arg_value: *const std::ffi::c_void
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
        ptr: *mut std::ffi::c_void,
        num_events_in_wait_list: u32,
        event_wait_list: *const cl_event,
        event: *mut cl_event
    ) -> cl_int;

    pub fn clFinish(command_queue: cl_command_queue) -> cl_int;
}

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

pub const CL_PLATFORM_PROFILE: cl_platform_info = 0x0900;
pub const CL_PLATFORM_VERSION: cl_platform_info = 0x0901;
pub const CL_PLATFORM_NAME: cl_platform_info = 0x0902;
pub const CL_PLATFORM_VENDOR: cl_platform_info = 0x0903;
pub const CL_PLATFORM_EXTENSIONS: cl_platform_info = 0x0904;

pub const CL_DEVICE_TYPE_CPU: cl_device_type = 1 << 1;
pub const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;

pub const CL_MEM_READ_WRITE: cl_mem_flags = 1 << 0;
pub const CL_MEM_WRITE_ONLY: cl_mem_flags = 1 << 1;
pub const CL_MEM_READ_ONLY: cl_mem_flags = 1 << 2;
pub const CL_MEM_COPY_HOST_PTR: cl_mem_flags = 1 << 5;

const MATRIX_SIZE: usize = 1024;
const WORK_GROUP_SIZE: usize = 16;

static KERNEL_SOURCE: &str = r#"
__kernel void matrix_multiply(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int size
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}
"#;

fn main() {
    unsafe {
        // Инициализация OpenCL
        let mut num_platforms = 0u32;
        clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms);
        
        let mut platforms = vec![std::ptr::null_mut(); num_platforms as usize];
        clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), &mut num_platforms);

        let mut selected_platform = None;
        let mut selected_device = std::ptr::null_mut();
        
        // Поиск GPU устройства
        'platform_loop: for platform in platforms.iter() {
            let mut num_devices = 0u32;
            let mut devices = Vec::new();
            
            let status = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 0, std::ptr::null_mut(), &mut num_devices);
            if status == 0 && num_devices > 0 {
                devices = vec![std::ptr::null_mut(); num_devices as usize];
                clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, num_devices, devices.as_mut_ptr(), &mut num_devices);
                
                selected_platform = Some(*platform);
                selected_device = devices[0];
                println!("Найдено GPU устройство");
                break 'platform_loop;
            }
        }

        if selected_platform.is_none() {
            println!("GPU устройство не найдено");
            return;
        }

        // Создание контекста и очереди команд
        let mut err = 0;
        let context = clCreateContext(
            std::ptr::null(),
            1,
            &selected_device,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut err
        );
        if err != 0 {
            println!("Ошибка создания контекста: {}", err);
            return;
        }

        let command_queue = clCreateCommandQueue(
            context,
            selected_device,
            0,
            &mut err
        );
        if err != 0 {
            println!("Ошибка создания очереди команд: {}", err);
            return;
        }

        // Создание и компиляция программы
        let source = KERNEL_SOURCE.as_ptr() as *const i8;
        let source_len = KERNEL_SOURCE.len();
        let program = clCreateProgramWithSource(
            context,
            1,
            &source,
            &source_len,
            &mut err
        );
        if err != 0 {
            println!("Ошибка создания программы: {}", err);
            return;
        }

        let build_status = clBuildProgram(
            program,
            1,
            &selected_device,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut()
        );
        if build_status != 0 {
            println!("Ошибка компиляции программы: {}", build_status);
            return;
        }

        // Создание ядра
        let kernel = clCreateKernel(
            program,
            "matrix_multiply\0".as_ptr() as *const i8,
            &mut err
        );
        if err != 0 {
            println!("Ошибка создания ядра: {}", err);
            return;
        }

        // Генерация тестовых данных
        let matrix_size = MATRIX_SIZE * MATRIX_SIZE;
        let mut a = vec![1.0f32; matrix_size];
        let mut b = vec![2.0f32; matrix_size];
        let mut c = vec![0.0f32; matrix_size];

        println!("\nВходная матрица A:");
        for i in 0..4 {
            for j in 0..4 {
                print!("{:.1} ", a[i * MATRIX_SIZE + j]);
            }
            println!("...");
        }
        println!("...\n");

        println!("Входная матрица B:");
        for i in 0..4 {
            for j in 0..4 {
                print!("{:.1} ", b[i * MATRIX_SIZE + j]);
            }
            println!("...");
        }
        println!("...\n");

        // Создание буферов
        let a_buffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (matrix_size * std::mem::size_of::<f32>()),
            a.as_mut_ptr() as *mut std::ffi::c_void,
            &mut err
        );
        if err != 0 {
            println!("Ошибка создания буфера A: {}", err);
            return;
        }

        let b_buffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (matrix_size * std::mem::size_of::<f32>()),
            b.as_mut_ptr() as *mut std::ffi::c_void,
            &mut err
        );
        if err != 0 {
            println!("Ошибка создания буфера B: {}", err);
            return;
        }

        let c_buffer = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY,
            (matrix_size * std::mem::size_of::<f32>()),
            std::ptr::null_mut(),
            &mut err
        );
        if err != 0 {
            println!("Ошибка создания буфера C: {}", err);
            return;
        }

        // Установка аргументов ядра
        let size = MATRIX_SIZE as i32;
        let mut status = clSetKernelArg(kernel, 0, std::mem::size_of::<cl_mem>(), &a_buffer as *const _ as *const std::ffi::c_void);
        if status != 0 {
            println!("Ошибка установки аргумента 0: {}", status);
            return;
        }
        
        status = clSetKernelArg(kernel, 1, std::mem::size_of::<cl_mem>(), &b_buffer as *const _ as *const std::ffi::c_void);
        if status != 0 {
            println!("Ошибка установки аргумента 1: {}", status);
            return;
        }
        
        status = clSetKernelArg(kernel, 2, std::mem::size_of::<cl_mem>(), &c_buffer as *const _ as *const std::ffi::c_void);
        if status != 0 {
            println!("Ошибка установки аргумента 2: {}", status);
            return;
        }
        
        status = clSetKernelArg(kernel, 3, std::mem::size_of::<i32>(), &size as *const _ as *const std::ffi::c_void);
        if status != 0 {
            println!("Ошибка установки аргумента 3: {}", status);
            return;
        }

        // Запуск ядра
        let global_work_size = [MATRIX_SIZE, MATRIX_SIZE];
        let local_work_size = [WORK_GROUP_SIZE, WORK_GROUP_SIZE];

        status = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            2,
            std::ptr::null(),
            global_work_size.as_ptr(),
            local_work_size.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null_mut()
        );
        if status != 0 {
            println!("Ошибка запуска ядра: {}", status);
            return;
        }

        // Ожидание завершения выполнения
        status = clFinish(command_queue);
        if status != 0 {
            println!("Ошибка ожидания завершения: {}", status);
            return;
        }

        // Чтение результата
        status = clEnqueueReadBuffer(
            command_queue,
            c_buffer,
            true,
            0,
            matrix_size * std::mem::size_of::<f32>(),
            c.as_mut_ptr() as *mut std::ffi::c_void,
            0,
            std::ptr::null(),
            std::ptr::null_mut()
        );
        if status != 0 {
            println!("Ошибка чтения результата: {}", status);
            return;
        }

        println!("Результирующая матрица C:");
        for i in 0..4 {
            for j in 0..4 {
                print!("{:.1} ", c[i * MATRIX_SIZE + j]);
            }
            println!("...");
        }
        println!("...\n");

        println!("Матричное умножение выполнено успешно");
    }
}

