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
        param_value: *mut std::ffi::c_void,
        param_value_size_ret: *mut usize
    ) -> cl_int;
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
#[allow(non_camel_case_types)]
pub type cl_program_build_info = u32;

pub const CL_DEVICE_TYPE_CPU: cl_device_type = 1 << 1;
pub const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;
pub const CL_MEM_READ_ONLY: cl_mem_flags = 1 << 0;
pub const CL_MEM_WRITE_ONLY: cl_mem_flags = 1 << 1;
pub const CL_MEM_COPY_HOST_PTR: cl_mem_flags = 1 << 5;
pub const CL_MEM_ALLOC_HOST_PTR: cl_mem_flags = 1 << 4;
pub const CL_PROGRAM_BUILD_LOG: cl_program_build_info = 0x1183;

const MATRIX_SIZE: usize = 1024;
const WORK_GROUP_SIZE: usize = 16;

// Тип матриц для вычислений
#[derive(PartialEq)]
enum MatrixType {
    OnesAndTwos,
    ThreesAndFours,
    Random
}

static KERNEL_SOURCE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void matrix_multiply(
    __global const double* a,
    __global const double* b,
    __global double* c,
    __local double* a_tile,
    __local double* b_tile,
    const int size
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int block_size = get_local_size(0);
    
    double sum = 0.0;
    
    // Количество блоков для обработки
    const int num_blocks = size / block_size;
    
    for (int block = 0; block < num_blocks; block++) {
        // Загрузка блоков в локальную память
        const int a_idx = row * size + block * block_size + local_col;
        const int b_idx = (block * block_size + local_row) * size + col;
        
        a_tile[local_row * block_size + local_col] = a[a_idx];
        b_tile[local_row * block_size + local_col] = b[b_idx];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Предварительный расчет базовых индексов
        const int a_row_offset = local_row * block_size;
        const int b_col_offset = local_col;
        
        // Развернутое умножение блоков для block_size = 16
        sum = fma(a_tile[a_row_offset + 0], b_tile[b_col_offset + 0 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 1], b_tile[b_col_offset + 1 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 2], b_tile[b_col_offset + 2 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 3], b_tile[b_col_offset + 3 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 4], b_tile[b_col_offset + 4 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 5], b_tile[b_col_offset + 5 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 6], b_tile[b_col_offset + 6 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 7], b_tile[b_col_offset + 7 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 8], b_tile[b_col_offset + 8 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 9], b_tile[b_col_offset + 9 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 10], b_tile[b_col_offset + 10 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 11], b_tile[b_col_offset + 11 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 12], b_tile[b_col_offset + 12 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 13], b_tile[b_col_offset + 13 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 14], b_tile[b_col_offset + 14 * block_size], sum);
        sum = fma(a_tile[a_row_offset + 15], b_tile[b_col_offset + 15 * block_size], sum);
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Сохранение результата
    if (row < size && col < size) {
        c[row * size + col] = sum;
    }
}
"#;

// Функция для инициализации матриц в зависимости от типа
fn initialize_matrices(matrix_type: MatrixType, size: usize) -> (Vec<f64>, Vec<f64>) {
    let matrix_elements = size * size;
    let (mut a, mut b) = match matrix_type {
        MatrixType::OnesAndTwos => {
            (vec![1.0f64; matrix_elements], vec![2.0f64; matrix_elements])
        },
        MatrixType::ThreesAndFours => {
            (vec![3.0f64; matrix_elements], vec![4.0f64; matrix_elements])
        },
        MatrixType::Random => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let a: Vec<f64> = (0..matrix_elements).map(|_| rng.gen_range(0.0..1.0)).collect();
            let b: Vec<f64> = (0..matrix_elements).map(|_| rng.gen_range(0.0..1.0)).collect();
            (a, b)
        }
    };
    (a, b)
}

// Функция для CPU реализации матричного умножения
fn cpu_matrix_multiply(a: &[f64], b: &[f64], c: &mut [f64], size: usize) {
    println!("\nНачало CPU вычислений для верификации...");
    let start_time = std::time::Instant::now();
    
    for i in 0..size {
        if i % 100 == 0 {
            println!("CPU: обработка строки {}/{}", i, size);
        }
        for j in 0..size {
            let mut sum = 0.0f64;
            for k in 0..size {
                sum += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = sum;
        }
    }
    
    let duration = start_time.elapsed();
    println!("CPU вычисления завершены за {:?}", duration);
    
    // Вывод результатов CPU вычислений
    println!("\nРезультирующая матрица C (CPU) ({}x{}):", size, size);
    for i in 0..4 {
        for j in 0..4 {
            print!("{:.1} ", c[i * size + j]);
        }
        println!("...");
    }
    println!("...\n");
}

// Функция для сравнения результатов GPU и CPU
fn compare_results(gpu_result: &[f64], cpu_result: &[f64], size: usize) -> bool {
    println!("\nСравнение результатов GPU и CPU...");
    let epsilon = 1e-10; // Уменьшенная погрешность для double
    let mut max_diff = 0.0f64;
    let mut diff_count = 0;
    
    for i in 0..size {
        for j in 0..size {
            let idx = i * size + j;
            let diff = (gpu_result[idx] - cpu_result[idx]).abs();
            if diff > epsilon {
                diff_count += 1;
                max_diff = max_diff.max(diff);
            }
        }
    }
    
    if diff_count > 0 {
        println!("Обнаружены расхождения:");
        println!("Количество различающихся элементов: {}", diff_count);
        println!("Максимальная разница: {}", max_diff);
        false
    } else {
        println!("Результаты GPU и CPU полностью совпадают!");
        true
    }
}

fn main() {
    unsafe {
        // Выбор типа матриц
        let matrix_type = MatrixType::Random; // Можно изменить на OnesAndTwos или ThreesAndFours или Random

        println!("Начало выполнения программы умножения матриц на GPU");
        println!("Размер матриц: {}x{}", MATRIX_SIZE, MATRIX_SIZE);
        println!("Размер рабочей группы: {}x{}", WORK_GROUP_SIZE, WORK_GROUP_SIZE);
        println!("\nИнициализация OpenCL...");
        
        // Инициализация OpenCL с обработкой ошибок
        let mut num_platforms = 0u32;
        let mut status = clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms);
        if status != 0 {
            println!("Ошибка при получении количества платформ OpenCL: {}", status);
            return;
        }
        
        println!("Найдено платформ OpenCL: {}", num_platforms);
        
        if num_platforms == 0 {
            println!("Не найдено платформ OpenCL");
            return;
        }
        
        let mut platforms = vec![std::ptr::null_mut(); num_platforms as usize];
        status = clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), &mut num_platforms);
        if status != 0 {
            println!("Ошибка при получении списка платформ OpenCL: {}", status);
            return;
        }

        let mut selected_platform = None;
        let mut selected_device = std::ptr::null_mut();
        
        println!("\nПоиск GPU устройства...");
        
        // Поиск GPU устройства с проверкой ошибок
        'platform_loop: for platform in platforms.iter() {
            let mut num_devices = 0u32;
            
            status = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 0, std::ptr::null_mut(), &mut num_devices);
            if status == 0 && num_devices > 0 {
                let mut devices = vec![std::ptr::null_mut(); num_devices as usize];
                status = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, num_devices, devices.as_mut_ptr(), &mut num_devices);
                if status == 0 {
                    selected_platform = Some(*platform);
                    selected_device = devices[0];
                    println!("Найдено GPU устройство");
                    break 'platform_loop;
                }
            }
        }

        if selected_platform.is_none() {
            println!("GPU устройство не найдено, пробуем CPU");
            for platform in platforms.iter() {
                let mut num_devices = 0u32;
                status = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_CPU, 0, std::ptr::null_mut(), &mut num_devices);
                if status == 0 && num_devices > 0 {
                    let mut devices = vec![std::ptr::null_mut(); num_devices as usize];
                    status = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_CPU, num_devices, devices.as_mut_ptr(), &mut num_devices);
                    if status == 0 {
                        selected_platform = Some(*platform);
                        selected_device = devices[0];
                        println!("Найдено CPU устройство");
                        break;
                    }
                }
            }
        }

        if selected_platform.is_none() {
            println!("Не найдено подходящих OpenCL устройств");
            return;
        }

        println!("\nСоздание контекста OpenCL...");
        
        // Создание контекста с проверкой ошибок
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
            println!("Ошибка при создании контекста OpenCL: {}", err);
            return;
        }

        println!("Создание очереди команд...");
        let command_queue = clCreateCommandQueue(context, selected_device, 0, &mut err);
        if err != 0 {
            println!("Ошибка при создании очереди команд: {}", err);
            clReleaseContext(context);
            return;
        }

        println!("\nКомпиляция OpenCL программы...");
        // Создание и компиляция программы с расширенной обработкой ошибок
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
            println!("Ошибка при создании программы: {}", err);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
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
            // Получение лога ошибок компиляции
            let mut log_size: usize = 0;
            clGetProgramBuildInfo(
                program,
                selected_device,
                CL_PROGRAM_BUILD_LOG,
                0,
                std::ptr::null_mut(),
                &mut log_size
            );

            let mut build_log = vec![0u8; log_size];
            clGetProgramBuildInfo(
                program,
                selected_device,
                CL_PROGRAM_BUILD_LOG,
                log_size,
                build_log.as_mut_ptr() as *mut std::ffi::c_void,
                std::ptr::null_mut()
            );

            println!("Ошибка при компиляции программы: {}", build_status);
            println!("Лог компиляции: {}", String::from_utf8_lossy(&build_log));
            
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        println!("Создание ядра OpenCL...");
        let kernel = clCreateKernel(
            program,
            "matrix_multiply\0".as_ptr() as *const i8,
            &mut err
        );
        if err != 0 {
            println!("Ошибка при создании ядра: {}", err);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        // Проверка размеров матриц на кратность 4 (из-за double4)
        if MATRIX_SIZE % 4 != 0 {
            println!("Размер матрицы должен быть кратен 4");
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        println!("\nПодготовка данных для умножения матриц...");
        // Инициализация матриц в зависимости от выбранного типа
        let matrix_elements = MATRIX_SIZE * MATRIX_SIZE;
        let (mut a, mut b) = initialize_matrices(matrix_type, MATRIX_SIZE);
        let mut c = vec![0.0f64; matrix_elements];

        // Создаем копии для CPU вычислений
        let mut cpu_a = a.clone();
        let mut cpu_b = b.clone();
        let mut cpu_c = vec![0.0f64; matrix_elements];

        println!("\nВходная матрица A ({}x{}):", MATRIX_SIZE, MATRIX_SIZE);
        for i in 0..4 {
            for j in 0..4 {
                print!("{:.1} ", a[i * MATRIX_SIZE + j]);
            }
            println!("...");
        }
        println!("...\n");

        println!("Входная матрица B ({}x{}):", MATRIX_SIZE, MATRIX_SIZE);
        for i in 0..4 {
            for j in 0..4 {
                print!("{:.1} ", b[i * MATRIX_SIZE + j]);
            }
            println!("...");
        }
        println!("...\n");

        let local_mem_size = (WORK_GROUP_SIZE * WORK_GROUP_SIZE * std::mem::size_of::<f64>()) as usize;
        println!("Размер локальной памяти для тайлов: {} байт", local_mem_size);

        println!("Создание буферов OpenCL...");
        // Создание буферов с правильными размерами
        let a_buffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
            matrix_elements * std::mem::size_of::<f64>(),
            a.as_mut_ptr() as *mut std::ffi::c_void,
            &mut err
        );
        if err != 0 {
            println!("Ошибка при создании буфера A: {}", err);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        let b_buffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
            matrix_elements * std::mem::size_of::<f64>(),
            b.as_mut_ptr() as *mut std::ffi::c_void,
            &mut err
        );
        if err != 0 {
            println!("Ошибка при создании буфера B: {}", err);
            clReleaseMemObject(a_buffer);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        let c_buffer = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
            matrix_elements * std::mem::size_of::<f64>(),
            std::ptr::null_mut(),
            &mut err
        );
        if err != 0 {
            println!("Ошибка при создании буфера C: {}", err);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(a_buffer);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        println!("Установка аргументов ядра...");
        // Установка аргументов ядра с проверкой каждого аргумента
        let mut status = 0;
        status |= clSetKernelArg(kernel, 0, std::mem::size_of::<cl_mem>(), &a_buffer as *const _ as *const std::ffi::c_void);
        status |= clSetKernelArg(kernel, 1, std::mem::size_of::<cl_mem>(), &b_buffer as *const _ as *const std::ffi::c_void);
        status |= clSetKernelArg(kernel, 2, std::mem::size_of::<cl_mem>(), &c_buffer as *const _ as *const std::ffi::c_void);
        status |= clSetKernelArg(kernel, 3, local_mem_size, std::ptr::null());
        status |= clSetKernelArg(kernel, 4, local_mem_size, std::ptr::null());
        status |= clSetKernelArg(kernel, 5, std::mem::size_of::<i32>(), &(MATRIX_SIZE as i32) as *const _ as *const std::ffi::c_void);

        if status != 0 {
            println!("Ошибка при установке аргументов ядра: {}", status);
            clReleaseMemObject(c_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(a_buffer);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        println!("\nЗапуск вычислений на GPU...");
        let gpu_start_time = std::time::Instant::now();
        
        let global_size = [MATRIX_SIZE, MATRIX_SIZE];
        let local_size = [WORK_GROUP_SIZE, WORK_GROUP_SIZE];

        status = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            2,
            std::ptr::null(),
            global_size.as_ptr(),
            local_size.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null_mut()
        );

        if status != 0 {
            println!("Ошибка при запуске ядра: {}", status);
            clReleaseMemObject(c_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(a_buffer);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        println!("Ожидание завершения GPU вычислений...");
        status = clFinish(command_queue);
        let gpu_duration = gpu_start_time.elapsed();
        println!("GPU вычисления завершены за {:?}", gpu_duration);

        if status != 0 {
            println!("Ошибка при ожидании завершения: {}", status);
            clReleaseMemObject(c_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(a_buffer);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        println!("Чтение результатов GPU...");
        status = clEnqueueReadBuffer(
            command_queue,
            c_buffer,
            true,
            0,
            matrix_elements * std::mem::size_of::<f64>(),
            c.as_mut_ptr() as *mut std::ffi::c_void,
            0,
            std::ptr::null(),
            std::ptr::null_mut()
        );

        if status != 0 {
            println!("Ошибка при чтении результата: {}", status);
        } else {
            println!("\nРезультирующая матрица C (GPU) ({}x{}):", MATRIX_SIZE, MATRIX_SIZE);
            for i in 0..4 {
                for j in 0..4 {
                    print!("{:.1} ", c[i * MATRIX_SIZE + j]);
                }
                println!("...");
            }
            println!("...\n");
        }

        // Выполняем CPU вычисления
        cpu_matrix_multiply(&cpu_a, &cpu_b, &mut cpu_c, MATRIX_SIZE);

        // Сравниваем результаты
        let results_match = compare_results(&c, &cpu_c, MATRIX_SIZE);

        println!("\nИтоговая статистика:");
        println!("Время выполнения на GPU: {:?}", gpu_duration);
        println!("Результаты GPU и CPU {}", if results_match { "совпадают" } else { "различаются" });

        println!("\nОсвобождение ресурсов OpenCL...");
        // Освобождаем ресурсы в правильном порядке
        clReleaseMemObject(c_buffer);
        clReleaseMemObject(b_buffer);
        clReleaseMemObject(a_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        println!("Программа завершена.");
    }
}

