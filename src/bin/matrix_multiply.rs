//! Тестирование производительности умножения матриц на GPU
use anyhow::{Context, Result};
use opencl_neural::neural::GpuEmbeddings;
use safetensors::SafeTensors;
use std::fs;

const MODEL_PATH: &str = "model.safetensors";

/// Структура для тестирования производительности
struct BertModel {
    embeddings: Vec<f32>,
    gpu_embeddings: Option<GpuEmbeddings>,
}

impl BertModel {
    fn from_safetensors(tensors: &SafeTensors) -> Result<Self> {
        let word_embeddings = tensors
            .tensor("bert.embeddings.word_embeddings.weight")
            .context("Не найдены word embeddings")?;

        let embeddings = unsafe {
            std::slice::from_raw_parts(
                word_embeddings.data().as_ptr() as *const f32,
                word_embeddings.data().len() / std::mem::size_of::<f32>()
            ).to_vec()
        };

        let gpu_embeddings = GpuEmbeddings::new(&embeddings, 768).ok();

        Ok(Self {
            embeddings,
            gpu_embeddings,
        })
    }

    fn forward(&self, input_ids: &[u32]) -> Vec<f32> {
        let mut output = Vec::new();
        let embedding_dim = 768;
        
        for &token_id in input_ids {
            let start_idx = (token_id as usize) * embedding_dim;
            let end_idx = start_idx + embedding_dim;
            
            if end_idx <= self.embeddings.len() {
                output.extend_from_slice(&self.embeddings[start_idx..end_idx]);
            }
        }

        output
    }

    fn forward_gpu(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        match &self.gpu_embeddings {
            Some(gpu) => gpu.forward(input_ids),
            None => Err(anyhow::anyhow!("GPU embeddings not initialized"))
        }
    }
}

fn main() -> Result<()> {
    println!("Тестирование производительности матричных операций\n");

    // Загружаем модель
    let model_data = fs::read(MODEL_PATH)
        .context("Не удалось прочитать файл модели")?;
    
    let tensors = SafeTensors::deserialize(&model_data)
        .context("Не удалось десериализовать модель")?;
    
    let model = BertModel::from_safetensors(&tensors)?;

    // Тестовые данные
    let input_ids: Vec<u32> = (0..100).collect(); // 100 последовательных токенов

    // Прогрев GPU
    println!("\nПрогрев GPU...");
    let _ = model.forward_gpu(&input_ids)?;

    // Тесты производительности
    const NUM_ITERATIONS: u32 = 100;
    println!("\nЗапуск {} итераций...", NUM_ITERATIONS);

    // Замер GPU
    println!("Вычисления на GPU...");
    let gpu_start = std::time::Instant::now();
    for _ in 0..NUM_ITERATIONS {
        let _ = model.forward_gpu(&input_ids)?;
    }
    let gpu_duration = gpu_start.elapsed();
    let gpu_avg = gpu_duration.as_secs_f64() / NUM_ITERATIONS as f64;
    
    // Замер CPU
    println!("Вычисления на CPU...");
    let cpu_start = std::time::Instant::now();
    for _ in 0..NUM_ITERATIONS {
        let _ = model.forward(&input_ids);
    }
    let cpu_duration = cpu_start.elapsed();
    let cpu_avg = cpu_duration.as_secs_f64() / NUM_ITERATIONS as f64;

    // Сравнение результатов
    let speedup = cpu_avg / gpu_avg;
    let improvement_percent = (speedup - 1.0) * 100.0;

    println!("\nРезультаты сравнения производительности:");
    println!("----------------------------------------");
    println!("Среднее время GPU: {:.6} мс", gpu_avg * 1000.0);
    println!("Среднее время CPU: {:.6} мс", cpu_avg * 1000.0);
    println!("GPU быстрее CPU в {:.2}x раз", speedup);
    println!("Улучшение производительности: {:.1}%", improvement_percent);
    
    // Проверка корректности
    let gpu_embeddings = model.forward_gpu(&input_ids)?;
    let cpu_embeddings = model.forward(&input_ids);
    
    println!("\nПроверка корректности:");
    println!("Размер выходных данных: {}", gpu_embeddings.len());
    println!("Первые 5 значений (GPU):");
    for &value in gpu_embeddings.iter().take(5) {
        print!("{:.4} ", value);
    }
    println!("\nПервые 5 значений (CPU):");
    for &value in cpu_embeddings.iter().take(5) {
        print!("{:.4} ", value);
    }
    println!();

    Ok(())
} 