//! Операции над матрицами

use super::types::MatrixType;
use rand::Rng;

/// Инициализирует матрицы заданного типа и размера
pub fn initialize_matrices(matrix_type: MatrixType, size: usize) -> (Vec<f64>, Vec<f64>) {
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

/// CPU реализация матричного умножения
pub fn cpu_matrix_multiply(a: &[f64], b: &[f64], c: &mut [f64], size: usize) {
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

/// Сравнивает результаты GPU и CPU вычислений
pub fn compare_results(gpu_result: &[f64], cpu_result: &[f64], size: usize) -> bool {
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