//! GPU-ускоренные операции с эмбеддингами

use crate::opencl::{types::*, bindings::*, cl_check, cl_create};
use anyhow::Result;
use std::ptr;
use anyhow::Context;

/// OpenCL ядро для получения эмбеддингов
// pub static EMBEDDING_KERNEL: &str = r#"
// __kernel void get_embeddings(
//     __global const float* embeddings_matrix,
//     __global const uint* input_ids,
//     __global float* output_embeddings,
//     const int embedding_dim,
//     const int sequence_length
// ) {
//     int gid = get_global_id(0);
//     if (gid >= sequence_length) return;
    
//     uint token_id = input_ids[gid];
//     int start_idx = token_id * embedding_dim;
    
//     // Векторизованное копирование
//     #pragma unroll 8
//     for (int i = 0; i < embedding_dim; i++) {
//         output_embeddings[gid * embedding_dim + i] = embeddings_matrix[start_idx + i];
//     }
// }
// "#;

// pub static EMBEDDING_KERNEL: &str = r#"
// __kernel void get_embeddings(
//     __global const float* embeddings_matrix,
//     __global const int* input_ids,  // Изменено с uint на int
//     __global float* output_embeddings,
//     const int embedding_dim,
//     const int sequence_length
// ) {
//     int gid = get_global_id(0);
//     if (gid >= sequence_length) return;
    
//     int token_id = input_ids[gid];  // Изменено с uint на int
//     int start_idx = token_id * embedding_dim;
    
//     // Векторизованное копирование
//     #pragma unroll 8
//     for (int i = 0; i < embedding_dim; i++) {
//         output_embeddings[gid * embedding_dim + i] = embeddings_matrix[start_idx + i];
//     }
// }
// "#;

pub static EMBEDDING_KERNEL: &str = r#"
__kernel void get_embeddings(
    __global const float* embeddings_matrix,
    __global const long* input_ids,  // Изменено на long (i64)
    __global float* output_embeddings,
    const int embedding_dim,
    const int sequence_length
) {
    int gid = get_global_id(0);
    if (gid >= sequence_length) return;
    
    long token_id = input_ids[gid];  // Изменено на long (i64)
    int start_idx = token_id * embedding_dim;
    
    // Векторизованное копирование
    #pragma unroll 8
    for (int i = 0; i < embedding_dim; i++) {
        output_embeddings[gid * embedding_dim + i] = embeddings_matrix[start_idx + i];
    }
}
"#;

pub static EUCLIDEAN_DISTANCE_KERNEL: &str = r#"
__kernel void euclidean_distance(
    __global const float* vec1,
    __global const float* vec2,
    __global float* result,
    const int length
) {
    int gid = get_global_id(0);
    if (gid >= length) return;

    float diff = vec1[gid] - vec2[gid];
    result[gid] = diff * diff;
}
"#;

/// Структура для GPU-ускоренных эмбеддингов
pub struct GpuEmbeddings {
    pub context: cl_context,
    pub command_queue: cl_command_queue,
    pub program: cl_program,
    pub kernel: cl_kernel,
    pub embeddings_buffer: cl_mem,
    pub input_buffer: cl_mem,
    pub output_buffer: cl_mem,
    pub embedding_dim: usize,
    pub max_sequence_length: usize,
}

impl GpuEmbeddings {
    /// Создает новый экземпляр с загруженными весами
    pub fn new(embeddings: &[f32], embedding_dim: usize) -> Result<Self> {
        unsafe {
            // Инициализация OpenCL
            let mut platform_ids = vec![ptr::null_mut(); 1];
            let mut num_platforms = 0;
            
            cl_check!(clGetPlatformIDs(1, platform_ids.as_mut_ptr(), &mut num_platforms))?;
            
            let platform = platform_ids[0];
            
            // Поиск GPU устройства
            let mut device_ids = vec![ptr::null_mut(); 1];
            let mut num_devices = 0;
            
            cl_check!(clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_GPU,
                1,
                device_ids.as_mut_ptr(),
                &mut num_devices
            ))?;
            
            let device = device_ids[0];
            
            // Проверяем максимальный размер рабочей группы
            let mut max_work_group_size = 0usize;
            cl_check!(clGetDeviceInfo(
                device,
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                std::mem::size_of::<usize>(),
                &mut max_work_group_size as *mut _ as *mut std::ffi::c_void,
                ptr::null_mut()
            ))?;
            
            println!("Max work group size: {}", max_work_group_size);
            
            // Создание контекста и очереди команд
            let context = cl_create!(clCreateContext(
                ptr::null(),
                1,
                &device,
                None,
                ptr::null_mut(),
                &mut 0
            ))?;
            
            let command_queue = cl_create!(clCreateCommandQueue(
                context,
                device,
                0,
                &mut 0
            ))?;
            
            // Компиляция программы
            let source = EMBEDDING_KERNEL.as_ptr() as *const i8;
            let length = EMBEDDING_KERNEL.len();
            
            let program = cl_create!(clCreateProgramWithSource(
                context,
                1,
                &source,
                &length,
                &mut 0
            ))?;
            
            if let Err(e) = cl_check!(clBuildProgram(
                program,
                1,
                &device,
                ptr::null(),
                None,
                ptr::null_mut()
            )) {
                // В случае ошибки выводим лог компиляции
                let mut log_size = 0;
                cl_check!(clGetProgramBuildInfo(
                    program,
                    device,
                    CL_PROGRAM_BUILD_LOG,
                    0,
                    ptr::null_mut(),
                    &mut log_size
                ))?;
                
                let mut log = vec![0u8; log_size];
                cl_check!(clGetProgramBuildInfo(
                    program,
                    device,
                    CL_PROGRAM_BUILD_LOG,
                    log_size,
                    log.as_mut_ptr() as *mut std::ffi::c_void,
                    ptr::null_mut()
                ))?;
                
                println!("OpenCL compilation error:\n{}", String::from_utf8_lossy(&log));
                return Err(e);
            }
            
            let kernel = cl_create!(clCreateKernel(
                program,
                "get_embeddings\0".as_ptr() as *const i8,
                &mut 0
            ))?;
            
            // Создаем буфер для матрицы эмбеддингов с оптимизированными флагами
            let embeddings_buffer = cl_create!(clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                embeddings.len() * std::mem::size_of::<f32>(),
                embeddings.as_ptr() as *mut std::ffi::c_void,
                &mut 0
            ))?;
            
            // Создаем постоянные буферы с запасом
            let max_sequence_length = 1024*1024; // или другое подходящее значение
            let max_output_size = max_sequence_length * embedding_dim;

            let input_buffer = cl_create!(clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,  // Буфер для чтения и записи
                max_sequence_length * std::mem::size_of::<i32>(),  // Изменено с u32 на i32
                ptr::null_mut(),
                &mut 0
            ))?;

            let output_buffer = cl_create!(clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                max_output_size * std::mem::size_of::<f32>(),
                ptr::null_mut(),
                &mut 0
            ))?;

            Ok(Self {
                context,
                command_queue,
                program,
                kernel,
                embeddings_buffer,
                input_buffer,
                output_buffer,
                embedding_dim,
                max_sequence_length,
            })
        }
    }

    /// Вычисляет Евклидово расстояние между двумя векторами
    pub fn euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        unsafe {
            let length = vec1.len();
            if length != vec2.len() {
                return Err(anyhow::anyhow!("Vectors must have the same length"));
            }

            // Создаем буферы для входных данных и результата
            let vec1_buffer = cl_create!(clCreateBuffer(
                self.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                length * std::mem::size_of::<f32>(),
                vec1.as_ptr() as *mut std::ffi::c_void,
                &mut 0
            ))?;

            let vec2_buffer = cl_create!(clCreateBuffer(
                self.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                length * std::mem::size_of::<f32>(),
                vec2.as_ptr() as *mut std::ffi::c_void,
                &mut 0
            ))?;

            let result_buffer = cl_create!(clCreateBuffer(
                self.context,
                CL_MEM_READ_WRITE,
                length * std::mem::size_of::<f32>(),
                ptr::null_mut(),
                &mut 0
            ))?;

            // Создаем ядро для вычисления Евклидова расстояния
            let kernel = cl_create!(clCreateKernel(
                self.program,
                "euclidean_distance\0".as_ptr() as *const i8,
                &mut 0
            ))?;

            // Устанавливаем аргументы ядра
            cl_check!(clSetKernelArg(
                kernel,
                0,
                std::mem::size_of::<cl_mem>(),
                &vec1_buffer as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                kernel,
                1,
                std::mem::size_of::<cl_mem>(),
                &vec2_buffer as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                kernel,
                2,
                std::mem::size_of::<cl_mem>(),
                &result_buffer as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                kernel,
                3,
                std::mem::size_of::<i32>(),
                &(length as i32) as *const _ as *const std::ffi::c_void
            ))?;

            // Запускаем ядро
            let global_work_size = length;
            let local_work_size = 128; // Оптимальный размер рабочей группы

            cl_check!(clEnqueueNDRangeKernel(
                self.command_queue,
                kernel,
                1,
                ptr::null(),
                &global_work_size,
                &local_work_size,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            // Читаем результат
            let mut result = vec![0.0f32; length];
            cl_check!(clEnqueueReadBuffer(
                self.command_queue,
                result_buffer,
                true,
                0,
                length * std::mem::size_of::<f32>(),
                result.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            // Вычисляем сумму квадратов разностей
            let sum_of_squares: f32 = result.iter().sum();

            // Освобождаем ресурсы
            clReleaseMemObject(vec1_buffer);
            clReleaseMemObject(vec2_buffer);
            clReleaseMemObject(result_buffer);
            clReleaseKernel(kernel);

            // Возвращаем квадратный корень из суммы
            Ok(sum_of_squares.sqrt())
        }
    }

    /// Получает эмбеддинги для последовательности токенов
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        unsafe {
            let sequence_length = input_ids.len();
            if sequence_length > self.max_sequence_length {
                return Err(anyhow::anyhow!("Sequence length exceeds maximum"));
            }

            let output_size = sequence_length * self.embedding_dim;
            let mut output = vec![0.0f32; output_size];

            // Копируем входные данные
            cl_check!(clEnqueueWriteBuffer(
                self.command_queue,
                self.input_buffer,
                false,
                0,
                input_ids.len() * std::mem::size_of::<u32>(),
                input_ids.as_ptr() as *const std::ffi::c_void,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            // Устанавливаем аргументы ядра (используем существующие буферы)
            cl_check!(clSetKernelArg(
                self.kernel,
                0,
                std::mem::size_of::<cl_mem>(),
                &self.embeddings_buffer as *const _ as *const std::ffi::c_void
            ))?;
            
            cl_check!(clSetKernelArg(
                self.kernel,
                1,
                std::mem::size_of::<cl_mem>(),
                &self.input_buffer as *const _ as *const std::ffi::c_void
            ))?;
            
            cl_check!(clSetKernelArg(
                self.kernel,
                2,
                std::mem::size_of::<cl_mem>(),
                &self.output_buffer as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                self.kernel,
                3,
                std::mem::size_of::<i32>(),
                &(self.embedding_dim as i32) as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                self.kernel,
                4,
                std::mem::size_of::<i32>(),
                &(sequence_length as i32) as *const _ as *const std::ffi::c_void
            ))?;

            // Запускаем ядро и читаем результат
            let local_work_size = 128;
            let global_work_size = (sequence_length + local_work_size - 1) / local_work_size * local_work_size;

            cl_check!(clEnqueueNDRangeKernel(
                self.command_queue,
                self.kernel,
                1,
                ptr::null(),
                &global_work_size,
                &local_work_size,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            cl_check!(clEnqueueReadBuffer(
                self.command_queue,
                self.output_buffer,
                true,
                0,
                output_size * std::mem::size_of::<f32>(),
                output.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            Ok(output)
        }
    }

    /// Получает эмбеддинги для последовательности токенов (с input_ids типа i32)
    pub fn forward_i32(&self, input_ids: &[i32]) -> Result<Vec<f32>> {
        unsafe {
            let sequence_length = input_ids.len();
            if sequence_length > self.max_sequence_length {
                return Err(anyhow::anyhow!("Sequence length exceeds maximum"));
            }

            let output_size = sequence_length * self.embedding_dim;
            let mut output = vec![0.0f32; output_size];

            // Копируем входные данные
            cl_check!(clEnqueueWriteBuffer(
                self.command_queue,
                self.input_buffer,
                false,
                0,
                input_ids.len() * std::mem::size_of::<i32>(),  // Изменено с u32 на i32
                input_ids.as_ptr() as *const std::ffi::c_void,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            // Устанавливаем аргументы ядра (используем существующие буферы)
            cl_check!(clSetKernelArg(
                self.kernel,
                0,
                std::mem::size_of::<cl_mem>(),
                &self.embeddings_buffer as *const _ as *const std::ffi::c_void
            ))?;
            
            cl_check!(clSetKernelArg(
                self.kernel,
                1,
                std::mem::size_of::<cl_mem>(),
                &self.input_buffer as *const _ as *const std::ffi::c_void
            ))?;
            
            cl_check!(clSetKernelArg(
                self.kernel,
                2,
                std::mem::size_of::<cl_mem>(),
                &self.output_buffer as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                self.kernel,
                3,
                std::mem::size_of::<i32>(),
                &(self.embedding_dim as i32) as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                self.kernel,
                4,
                std::mem::size_of::<i32>(),
                &(sequence_length as i32) as *const _ as *const std::ffi::c_void
            ))?;

            // Запускаем ядро и читаем результат
            let local_work_size = 128;
            let global_work_size = (sequence_length + local_work_size - 1) / local_work_size * local_work_size;

            cl_check!(clEnqueueNDRangeKernel(
                self.command_queue,
                self.kernel,
                1,
                ptr::null(),
                &global_work_size,
                &local_work_size,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            cl_check!(clEnqueueReadBuffer(
                self.command_queue,
                self.output_buffer,
                true,
                0,
                output_size * std::mem::size_of::<f32>(),
                output.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            Ok(output)
        }
    }

    /// Получает эмбеддинги для последовательности токенов (с input_ids типа i64)
    pub fn forward_i64(&self, input_ids: &[i64]) -> Result<Vec<f32>> {
        unsafe {
            let sequence_length = input_ids.len();
            if sequence_length > self.max_sequence_length {
                return Err(anyhow::anyhow!("Sequence length exceeds maximum"));
            }

            let output_size = sequence_length * self.embedding_dim;
            let mut output = vec![0.0f32; output_size];

            // Копируем входные данные
            cl_check!(clEnqueueWriteBuffer(
                self.command_queue,
                self.input_buffer,
                false,
                0,
                input_ids.len() * std::mem::size_of::<i64>(),  // Изменено на i64
                input_ids.as_ptr() as *const std::ffi::c_void,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            // Устанавливаем аргументы ядра (используем существующие буферы)
            cl_check!(clSetKernelArg(
                self.kernel,
                0,
                std::mem::size_of::<cl_mem>(),
                &self.embeddings_buffer as *const _ as *const std::ffi::c_void
            ))?;
            
            cl_check!(clSetKernelArg(
                self.kernel,
                1,
                std::mem::size_of::<cl_mem>(),
                &self.input_buffer as *const _ as *const std::ffi::c_void
            ))?;
            
            cl_check!(clSetKernelArg(
                self.kernel,
                2,
                std::mem::size_of::<cl_mem>(),
                &self.output_buffer as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                self.kernel,
                3,
                std::mem::size_of::<i32>(),
                &(self.embedding_dim as i32) as *const _ as *const std::ffi::c_void
            ))?;

            cl_check!(clSetKernelArg(
                self.kernel,
                4,
                std::mem::size_of::<i32>(),
                &(sequence_length as i32) as *const _ as *const std::ffi::c_void
            ))?;

            // Запускаем ядро и читаем результат
            let local_work_size = 128;
            let global_work_size = (sequence_length + local_work_size - 1) / local_work_size * local_work_size;

            cl_check!(clEnqueueNDRangeKernel(
                self.command_queue,
                self.kernel,
                1,
                ptr::null(),
                &global_work_size,
                &local_work_size,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            cl_check!(clEnqueueReadBuffer(
                self.command_queue,
                self.output_buffer,
                true,
                0,
                output_size * std::mem::size_of::<f32>(),
                output.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                ptr::null(),
                ptr::null_mut()
            ))?;

            Ok(output)
        }
    }
}

impl Drop for GpuEmbeddings {
    fn drop(&mut self) {
        unsafe {
            clReleaseMemObject(self.embeddings_buffer);
            clReleaseKernel(self.kernel);
            clReleaseProgram(self.program);
            clReleaseCommandQueue(self.command_queue);
            clReleaseContext(self.context);
        }
    }
} 