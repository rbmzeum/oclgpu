//! Пример использования предобученной модели BERT для классификации текста
use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use safetensors::SafeTensors;
use std::fs;
use std::path::Path;
use std::time::Duration;
use tokenizers::Tokenizer;
use opencl_neural::neural::GpuEmbeddings;
use std::fmt::Write;

const MODEL_PATH: &str = "model.safetensors";
const TOKENIZER_URL: &str = "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json";
const TOKENIZER_PATH: &str = "tokenizer.json";

/// Структура для хранения параметров BERT модели
struct BertModel {
    embeddings: Vec<f32>,
    attention_weights: Vec<f32>,
    layer_norm_weights: Vec<f32>,
    gpu_embeddings: Option<GpuEmbeddings>,
}

impl BertModel {
    /// Загружает веса модели из safetensors файла
    fn from_safetensors(tensors: &SafeTensors) -> Result<Self> {
        // Выведем список доступных тензоров для диагностики
        println!("\nДоступные тензоры в модели:");
        for name in tensors.names() {
            println!("  - {}", name);
        }
        println!();

        // Используем правильные имена тензоров из модели
        let word_embeddings = tensors
            .tensor("bert.embeddings.word_embeddings.weight")
            .context("Не найдены word embeddings")?;
        
        let attention = tensors
            .tensor("bert.encoder.layer.0.attention.self.query.weight")
            .context("Не найдены attention weights")?;
            
        let layer_norm = tensors
            .tensor("bert.encoder.layer.0.attention.output.LayerNorm.gamma")
            .or_else(|_| tensors.tensor("bert.embeddings.LayerNorm.gamma"))
            .context("Не найдены layer norm weights")?;

        println!("\nРазмерности тензоров:");
        println!("word_embeddings: {:?}", word_embeddings.shape());
        println!("attention: {:?}", attention.shape());
        println!("layer_norm: {:?}", layer_norm.shape());

        // Преобразуем тензоры в векторы f32
        let embeddings = unsafe {
            std::slice::from_raw_parts(
                word_embeddings.data().as_ptr() as *const f32,
                word_embeddings.data().len() / std::mem::size_of::<f32>()
            ).to_vec()
        };

        let attention_weights = unsafe {
            std::slice::from_raw_parts(
                attention.data().as_ptr() as *const f32,
                attention.data().len() / std::mem::size_of::<f32>()
            ).to_vec()
        };

        let layer_norm_weights = unsafe {
            std::slice::from_raw_parts(
                layer_norm.data().as_ptr() as *const f32,
                layer_norm.data().len() / std::mem::size_of::<f32>()
            ).to_vec()
        };

        println!("\nРазмеры векторов:");
        println!("embeddings: {}", embeddings.len());
        println!("attention_weights: {}", attention_weights.len());
        println!("layer_norm_weights: {}", layer_norm_weights.len());

        let gpu_embeddings = GpuEmbeddings::new(&embeddings, 768).ok();

        Ok(Self {
            embeddings,
            attention_weights,
            layer_norm_weights,
            gpu_embeddings,
        })
    }

    /// Простой forward pass для демонстрации
    fn forward(&self, input_ids: &[u32]) -> Vec<f32> {
        // Получаем эмбеддинги для входных токенов
        let mut output = Vec::new();
        let embedding_dim = 768; // Стандартный размер для BERT base
        
        for &token_id in input_ids {
            let start_idx = (token_id as usize) * embedding_dim;
            let end_idx = start_idx + embedding_dim;
            
            if end_idx <= self.embeddings.len() {
                output.extend_from_slice(&self.embeddings[start_idx..end_idx]);
            }
        }

        output
    }

    /// GPU-ускоренный forward pass
    fn forward_gpu(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        if let Some(gpu_embeddings) = &self.gpu_embeddings {
            gpu_embeddings.forward(input_ids)
        } else {
            Err(anyhow!("GPU embeddings not initialized"))
        }
    }
}

/// Загружает или скачивает токенизатор
fn get_tokenizer() -> Result<Tokenizer> {
    if !Path::new(TOKENIZER_PATH).exists() {
        println!("Токенизатор не найден, загружаем из интернета...");
        
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("Не удалось создать HTTP клиент")?;

        let response = client.get(TOKENIZER_URL)
            .send()
            .context("Не удалось загрузить токенизатор")?;
            
        let bytes = response.bytes()
            .context("Не удалось получить байты токенизатора")?;
            
        fs::write(TOKENIZER_PATH, bytes)
            .context("Не удалось сохранить токенизатор")?;
        
        println!("Токенизатор успешно загружен и сохранен в {}", TOKENIZER_PATH);
    } else {
        println!("Используем локальный токенизатор из {}", TOKENIZER_PATH);
    }

    // Преобразуем ошибку в anyhow::Error
    Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| anyhow!("Не удалось загрузить токенизатор из файла: {}", e))
}

/// Тестовые примеры для BERT токенизатора
const TEST_CASES: &[&str] = &[
    "Hello world",
    "Machine learning is fascinating",
    "OpenCL GPU acceleration",
    "BERT embeddings test case",
    "[CLS] Special tokens [SEP]",
    "Тест юникода и спецсимволов !@#$%",
    "Multi-word hyphenated-text example",
    "Short",
    "This is a much longer sentence that should test the tokenizer's ability to handle sequences of varying length and complexity",
];

fn run_tokenizer_tests(model: &BertModel, tokenizer: &Tokenizer) -> Result<()> {
    println!("\nЗапуск тестов токенизатора BERT:");
    println!("================================");

    for (i, &test_case) in TEST_CASES.iter().enumerate() {
        println!("\nТест #{}", i + 1);
        println!("Входной текст: {}", test_case);

        // Токенизация
        let encoding = tokenizer.encode(test_case, true)
            .map_err(|e| anyhow!("Ошибка токенизации: {}", e))?;

        let tokens = encoding.get_tokens();
        let ids = encoding.get_ids();

        println!("Токены: {:?}", tokens);
        println!("ID токенов: {:?}", ids);

        // Получаем эмбеддинги через GPU
        let gpu_embeddings = model.forward_gpu(ids)?;

        // Статистика
        println!("Количество токенов: {}", tokens.len());
        println!("Размер эмбеддингов: {} байт", gpu_embeddings.len() * 4);
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("Тестирование BERT токенизатора и эмбеддингов\n");

    // Загружаем токенизатор
    let tokenizer = get_tokenizer()?;

    // Загружаем модель
    let model_data = fs::read(MODEL_PATH)
        .context("Не удалось прочитать файл модели")?;
    
    let tensors = SafeTensors::deserialize(&model_data)
        .context("Не удалось десериализовать модель")?;
    
    let model = BertModel::from_safetensors(&tensors)?;

    // Запускаем тесты
    run_tokenizer_tests(&model, &tokenizer)?;

    // Прогрев GPU и тесты производительности
    println!("\nТесты производительности:");
    println!("========================");

    // Создадим более длинный текст для лучшего сравнения
    let text = "Hello, I love machine learning! Neural networks are fascinating \
                and transformers have revolutionized natural language processing. \
                The ability to process and understand human language is one of the \
                most impressive achievements in artificial intelligence.";
    println!("Входной текст: {}", text);

    let encoding = tokenizer.encode(text, true)
        .map_err(|e| anyhow!("Ошибка токенизации: {}", e))?;
    
    println!("Количество токенов: {}", encoding.get_ids().len());

    // Прогрев GPU (первый запуск часто медленнее)
    println!("\nПрогрев GPU...");
    let _ = model.forward_gpu(&encoding.get_ids())?;

    // Несколько итераций для более точного замера
    const NUM_ITERATIONS: u32 = 100;
    println!("\nЗапуск {} итераций...", NUM_ITERATIONS);

    // Замер GPU
    println!("Вычисления на GPU...");
    let gpu_start = std::time::Instant::now();
    for _ in 0..NUM_ITERATIONS {
        let _ = model.forward_gpu(&encoding.get_ids())?;
    }
    let gpu_duration = gpu_start.elapsed();
    let gpu_avg = gpu_duration.as_secs_f64() / NUM_ITERATIONS as f64;
    
    // Замер CPU
    println!("Вычисления на CPU...");
    let cpu_start = std::time::Instant::now();
    for _ in 0..NUM_ITERATIONS {
        let _ = model.forward(&encoding.get_ids());
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
    
    // Проверка корректности результатов
    let gpu_embeddings = model.forward_gpu(&encoding.get_ids())?;
    let cpu_embeddings = model.forward(&encoding.get_ids());
    
    println!("\nПроверка корректности:");
    println!("Размер выходных эмбеддингов: {}", gpu_embeddings.len());
    println!("Первые 5 значений эмбеддингов (GPU):");
    for &value in gpu_embeddings.iter().take(5) {
        print!("{:.4} ", value);
    }
    println!("\nПервые 5 значений эмбеддингов (CPU):");
    for &value in cpu_embeddings.iter().take(5) {
        print!("{:.4} ", value);
    }
    println!();

    Ok(())
} 