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
use tch::{Tensor, Device, Kind};

const MODEL_URL: &str = "https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/model.safetensors";
const TOKENIZER_URL: &str = "https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/tokenizer.json";
const CONFIG_URL: &str = "https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/config.json";
const MODEL_PATH: &str = "modernbert_model.safetensors";
const TOKENIZER_PATH: &str = "modernbert_tokenizer.json";
const CONFIG_PATH: &str = "modernbert_config.json";

/// Загружает файл из интернета если он не существует локально
fn download_if_needed(url: &str, path: &str) -> Result<()> {
    if !Path::new(path).exists() {
        println!("Загрузка {}...", path);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;
        
        let mut response = client.get(url)
            .send()
            .context("Не удалось получить ответ от сервера")?;
        
        let total_size = response.content_length().unwrap_or(0);
        
        let pb = ProgressBar::new(total_size);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        let mut file = std::fs::File::create(path)
            .context("Не удалось создать файл")?;

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

struct ModernBertModel {
    tokenizer: Tokenizer,
    embeddings: GpuEmbeddings,
    config: serde_json::Value,
}

impl ModernBertModel {
    fn new() -> Result<Self> {
        // Загружаем необходимые файлы
        download_if_needed(MODEL_URL, MODEL_PATH)?;
        download_if_needed(TOKENIZER_URL, TOKENIZER_PATH)?;
        download_if_needed(CONFIG_URL, CONFIG_PATH)?;
        
        // Загружаем конфигурацию
        let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(CONFIG_PATH)?)?;
        
        // Загружаем токенизатор
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow::anyhow!("Не удалось загрузить токенизатор: {}", e))?;

        // Загружаем веса модели
        let model_data: Vec<u8> = fs::read(MODEL_PATH)?;
        let tensors = SafeTensors::deserialize(&model_data)?;

        // Отладочная информация: выводим все имена тензоров
        println!("Доступные тензоры в модели:");
        for name in tensors.names() {
            println!("- {}", name);
        }
        
        // Получаем размерность эмбеддингов из конфигурации
        let hidden_size: usize = config["hidden_size"]
            .as_u64()
            .context("Не найден параметр hidden_size в конфигурации")? as usize;
            
        // Получаем эмбеддинги
        let embeddings_tensor = tensors
            .tensor("model.embeddings.tok_embeddings.weight")  // Используем правильное имя
            .context("Не найдены word embeddings")?;

        let embeddings_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                embeddings_tensor.data().as_ptr() as *const f32,
                embeddings_tensor.data().len() / std::mem::size_of::<f32>()
            )
        };

        let embeddings = GpuEmbeddings::new(embeddings_data, hidden_size)?;

        Ok(Self {
            tokenizer,
            embeddings,
            config,
        })
    }

    /// Преобразует тексты в эмбеддинги
    fn text_to_vector(&self, texts: &[&str]) -> Result<(Tensor, Tensor)> {
        // Токенизация текстов
        let encodings = self.tokenizer.encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Ошибка токенизации: {}", e))?;
    
        // Получаем максимальную длину последовательности для паддинга
        let max_len: usize = encodings.iter().map(|enc| enc.get_ids().len()).max().unwrap_or(0);
        
        // Создаем тензоры для input_ids и attention_mask
        let mut input_ids: Vec<i64> = Vec::new();
        let mut attention_mask: Vec<i64> = Vec::new();
        
        for encoding in encodings {
            let ids: &[u32] = encoding.get_ids();
            let mut padded_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let mask: Vec<i64> = vec![1; ids.len()];
            let mut padded_mask: Vec<i64> = mask.clone();
            
            // Паддинг до максимальной длины
            while padded_ids.len() < max_len {
                padded_ids.push(0);
                padded_mask.push(0);
            }
            
            input_ids.extend_from_slice(&padded_ids);
            attention_mask.extend_from_slice(&padded_mask);
        }
    
        // Преобразуем в тензоры PyTorch
        let device = Device::Cpu;
        
        let mask_tensor = Tensor::from_slice(&attention_mask)
            .reshape(&[texts.len() as i64, max_len as i64])
            .to_device(device);
    
        // Получаем эмбеддинги через GPU
        let hidden_states: Vec<f32> = self.embeddings.forward_i64(input_ids.as_slice())?;
    
        // Преобразуем Vec<f32> в Tensor
        let hidden_states_tensor = Tensor::from_slice(&hidden_states);
    
        // Убедимся, что hidden_states имеет размерность [batch_size, seq_len, hidden_size]
        let hidden_states_reshaped = hidden_states_tensor.reshape(&[texts.len() as i64, max_len as i64, -1]);
    
        Ok((hidden_states_reshaped, mask_tensor))
    }

    /// Применяет average pooling к эмбеддингам
    fn average_pooling(&self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor {
        println!("hidden_states size: {:?}", hidden_states.size());
        println!("attention_mask size: {:?}", attention_mask.size());
    
        let input_mask_expanded = attention_mask
            .unsqueeze(-1)
            .expand_as(&hidden_states)
            .to_dtype(Kind::Float, false, true);
    
        println!("input_mask_expanded size: {:?}", input_mask_expanded.size());
    
        let sum_embeddings = (&hidden_states * &input_mask_expanded)
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
        
        println!("sum_embeddings size: {:?}", sum_embeddings.size());
    
        let sum_mask = input_mask_expanded
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
        
        println!("sum_mask size: {:?}", sum_mask.size());
    
        let sum_mask = sum_mask.clamp(1e-9, f64::INFINITY);
        
        sum_embeddings / sum_mask
    }
}

fn main() -> Result<()> {
    println!("Инициализация ModernBERT модели...");
    let model = ModernBertModel::new()?;

    let test_texts = [
        "Hello, how are you?",
        "Bonjour, comment ça va?",
        "Hola, ¿cómo estás?",
        "Привет, как дела?",
        "你好，你好吗？",
    ];

    println!("\nПреобразование текстов в эмбеддинги...");
    let (hidden_states, attention_mask) = model.text_to_vector(&test_texts)?;
    
    println!("Применение Average Pooling...");
    let pooled_embeddings = model.average_pooling(hidden_states, attention_mask);
    
    println!("\nРезультаты:");
    println!("Размер эмбеддингов: {:?}", pooled_embeddings.size());
    println!("Эмбеддинги: {}", pooled_embeddings);

    Ok(())
}
