//! Генерация эмбеддингов с помощью E5 multilingual модели
use anyhow::{Context, Result};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use std::fs;
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use opencl_neural::{neural::GpuEmbeddings, cl_check, cl_create};
use std::io::Write;
use reqwest::blocking::Client;
use std::io::Read;

const MODEL_URL: &str = "https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/model.safetensors";
const TOKENIZER_URL: &str = "https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/tokenizer.json";
const MODEL_PATH: &str = "e5_large_multilingual.safetensors";
const TOKENIZER_PATH: &str = "e5_tokenizer.json";
const EMBEDDING_DIM: usize = 1024; // Размерность для E5-large

/// Загружает файл из интернета если он не существует локально
fn download_if_needed(url: &str, path: &str) -> Result<()> {
    if !Path::new(path).exists() {
        println!("Загрузка {}...", path);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(600)) // Увеличиваем таймаут
            .build()?;
        
        // Получаем ответ
        let mut response = client.get(url)
            .send()
            .context("Не удалось получить ответ от сервера")?;
        
        let total_size = response.content_length().unwrap_or(0);
        
        // Создаем прогресс-бар
        let pb = ProgressBar::new(total_size);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        // Открываем файл для записи
        let mut file = std::fs::File::create(path)
            .context("Не удалось создать файл")?;

        // Читаем данные чанками с отображением прогресса
        let mut downloaded: u64 = 0;
        
        loop {
            let mut buffer = vec![0; 8192];
            let bytes_read = response.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;
            pb.set_position(downloaded);
        }
            
        pb.finish_with_message("Загрузка завершена");
    }
    Ok(())
}

/// Структура для работы с E5 моделью
struct E5Model {
    tokenizer: Tokenizer,
    embeddings: GpuEmbeddings,
}

impl E5Model {
    /// Загружает модель и токенизатор
    fn new() -> Result<Self> {
        // Загружаем файлы если нужно
        download_if_needed(MODEL_URL, MODEL_PATH)?;
        download_if_needed(TOKENIZER_URL, TOKENIZER_PATH)?;
        
        // Загружаем токенизатор
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow::anyhow!("Не удалось загрузить токенизатор: {}", e))?;

        // Загружаем веса модели
        let model_data = fs::read(MODEL_PATH)
            .context("Не удалось прочитать файл модели")?;
        
        let tensors = SafeTensors::deserialize(&model_data)
            .context("Не удалось десериализовать модель")?;

        // Выводим доступные тензоры для отладки
        println!("Доступные тензоры в модели:");
        for tensor_name in tensors.names() {
            println!("- {}", tensor_name);
        }

        // Получаем эмбеддинги (используем правильный путь для E5)
        let embeddings_tensor = tensors
            .tensor("embeddings.word_embeddings.weight")  // Изменили путь к тензору
            .or_else(|_| tensors.tensor("model.embeddings.word_embeddings.weight"))  // Альтернативный путь
            .or_else(|_| tensors.tensor("e5.embeddings.word_embeddings.weight"))  // Еще один вариант
            .context("Не найдены word embeddings. Проверьте структуру модели.")?;

        let embeddings_data = unsafe {
            std::slice::from_raw_parts(
                embeddings_tensor.data().as_ptr() as *const f32,
                embeddings_tensor.data().len() / std::mem::size_of::<f32>()
            )
        };

        let embeddings = GpuEmbeddings::new(embeddings_data, EMBEDDING_DIM)?;

        Ok(Self {
            tokenizer,
            embeddings,
        })
    }

    /// Генерирует эмбеддинги для текста
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        // Подготавливаем текст в формате E5
        let prepared_text = format!("passage: {}", text);
        
        // Токенизируем
        let encoding = self.tokenizer.encode(
            prepared_text,
            true
        ).map_err(|e| anyhow::anyhow!("Ошибка токенизации: {}", e))?;

        // Получаем эмбеддинги через GPU для каждого токена
        let token_embeddings = self.embeddings.forward(encoding.get_ids())?;
        
        // Преобразуем в матрицу [num_tokens x embedding_dim]
        let num_tokens = encoding.get_ids().len();
        let token_matrix: Vec<Vec<f32>> = token_embeddings
            .chunks(EMBEDDING_DIM)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Выполняем average pooling по всем токенам
        let mut pooled_embedding = vec![0.0; EMBEDDING_DIM];
        for token_vector in token_matrix.iter() {
            for (i, &value) in token_vector.iter().enumerate() {
                pooled_embedding[i] += value;
            }
        }
        
        // Делим на количество токенов для усреднения
        for value in pooled_embedding.iter_mut() {
            *value /= num_tokens as f32;
        }

        // L2 нормализация
        let mut norm = 0.0;
        for &x in pooled_embedding.iter() {
            norm += x * x;
        }
        norm = norm.sqrt();
        
        let normalized = pooled_embedding.iter()
            .map(|&x| x / norm)
            .collect();

        Ok(normalized)
    }
}

fn f32_to_hex_string(value: f32) -> String {
    // Преобразуем f32 в u32, чтобы получить битовое представление
    let bits = value.to_bits();
    // Форматируем u32 в шестнадцатеричную строку без префикса "0x"
    format!("{:08X}", bits)
}

fn f32_vector_to_hex_string(vector: &[f32]) -> String {
    vector.iter()
        .map(|&value| f32_to_hex_string(value))
        .collect::<String>()
}

fn print_hex_vector(vector: &[f32]) -> String {
    let mut hex_string = String::new();
    
    for &value in vector.iter() {
        // Преобразуем значение в шестнадцатеричный формат и добавляем его к строке
        hex_string.push_str(&format!("{:02x}", (value * 100.0) as u8));
    }
    
    hex_string
}

fn main() -> Result<()> {
    println!("Инициализация E5 multilingual модели...");
    let model = E5Model::new()?;

    let test_texts = [
        "Искусственный интеллект становится важной частью нашей жизни",
        "Machine learning is transforming the world",
        "深度学习正在改变世界",
        "L'intelligence artificielle change notre façon de vivre",
        "La inteligencia artificial está cambiando nuestras vidas",
    ];

    println!("\nГенерация эмбеддингов для текстов на разных языках:");
    let mut embeddings = Vec::new();

    for (i, text) in test_texts.iter().enumerate() {
        println!("\nТекст {}: {}", i + 1, text);
        
        let embedding = model.encode(text)?;
        println!("Размер вектора: {}", embedding.len());
        
        // Компактный вывод вектора
        print!("Вектор: [");
        for (j, &value) in embedding.iter().enumerate() {
            if j < 10 || j > embedding.len() - 5 {
                // Выводим первые 10 и последние 4 значения
                print!("{:.3}", value);
                if j < embedding.len() - 1 {
                    print!(", ");
                }
            } else if j == 10 {
                print!("..., ");
            }
        }
        println!("]");

        // Компактный вывод вектора
        print!("Вектор (hex): [{:?}]", &f32_vector_to_hex_string(&embedding));
        
        // Статистика по вектору
        let min = embedding.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = embedding.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
        println!("Статистика: min={:.3}, max={:.3}, mean={:.3}", min, max, mean);
        
        embeddings.push(embedding);
    }

    Ok(())
} 