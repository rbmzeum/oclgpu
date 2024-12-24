//! OpenCL ядра для матричных операций

/// Исходный код ядра для матричного умножения
pub static MATRIX_MULTIPLY_KERNEL: &str = r#"
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