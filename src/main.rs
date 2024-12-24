//! Пример использования библиотеки

use opencl_neural::{
    matrix::{
        MatrixType,
        cpu_matrix_multiply,
        compare_results,
        initialize_matrices,
        MATRIX_MULTIPLY_KERNEL,
    },
    opencl::{types::*, bindings::*},
    utils::measure_time,
};

const MATRIX_SIZE: usize = 1024;
const WORK_GROUP_SIZE: usize = 16;

fn main() {
    unsafe {
        // Выбор типа матриц
        let matrix_type = MatrixType::Random;

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

        // Поиск GPU устройства
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

        // Если GPU не найден, пробуем CPU
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
        
        // Создание контекста
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
        
        // Создание и компиляция программы
        let source = MATRIX_MULTIPLY_KERNEL.as_ptr() as *const i8;
        let source_len = MATRIX_MULTIPLY_KERNEL.len();
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

        // Проверка размеров матриц
        if MATRIX_SIZE % 4 != 0 {
            println!("Размер матрицы должен быть кратен 4");
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        println!("\nПодготовка данных для умножения матриц...");
        
        // Инициализация матриц
        let matrix_elements = MATRIX_SIZE * MATRIX_SIZE;
        let (mut a, mut b) = initialize_matrices(matrix_type, MATRIX_SIZE);
        let mut c = vec![0.0f64; matrix_elements];

        // Копии для CPU вычислений
        let cpu_a = a.clone();
        let cpu_b = b.clone();
        let mut cpu_c = vec![0.0f64; matrix_elements];

        // Вывод входных данных
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
        
        // Создание буферов
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
        
        // Установка аргументов ядра
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

        // CPU вычисления для сравнения
        println!("\nЗапуск вычислений на CPU...");
        let (_, cpu_duration) = measure_time(|| {
            cpu_matrix_multiply(&cpu_a, &cpu_b, &mut cpu_c, MATRIX_SIZE);
        });

        // Сравнение результатов
        let results_match = compare_results(&c, &cpu_c, MATRIX_SIZE);

        // Статистика
        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
        let percentage_faster = (speedup - 1.0) * 100.0;

        println!("\nИтоговая статистика:");
        println!("Время выполнения на GPU: {:?}", gpu_duration);
        println!("Время выполнения на CPU: {:?}", cpu_duration);
        println!("GPU быстрее CPU в {:.2} раз", speedup);
        println!("GPU быстрее CPU на {:.2}%", percentage_faster);
        println!("Результаты GPU и CPU {}", if results_match { "совпадают" } else { "различаются" });

        println!("\nОсвобождение ресурсов OpenCL...");
        // Освобождение ресурсов
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

