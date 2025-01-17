//! Генерация кода на Rust с помощью Qwen2.5 Coder модели
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use std::fs;
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use opencl_neural::{neural::GpuEmbeddings, cl_check, cl_create};
use opencl_neural::opencl::clCreateBuffer;
use opencl_neural::opencl::clReleaseMemObject;
use opencl_neural::opencl::clEnqueueReadBuffer;
use opencl_neural::opencl::clEnqueueWriteBuffer;
use opencl_neural::opencl::clSetKernelArg;
use opencl_neural::opencl::clEnqueueNDRangeKernel;
use opencl_neural::cl_mem;
use opencl_neural::CL_MEM_READ_WRITE;
use serde_json::Value;
use std::ptr;
use std::io::{Write, Read, Seek, SeekFrom};
use std::fs::{File, OpenOptions};
use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use reqwest::header::{RANGE, CONTENT_LENGTH, CONTENT_RANGE};


// URLs для загрузки модели
const MODEL_URL_PART1: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/model-00001-of-00002.safetensors";
const MODEL_URL_PART2: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/model-00002-of-00002.safetensors";
const TOKENIZER_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/tokenizer.json";
const CONFIG_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/config.json";
const TOKENIZER_CONFIG_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/tokenizer_config.json";
const VOCAB_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/vocab.json";
const MERGES_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/merges.txt";

// Пути для сохранения файлов
const MODEL_PATH_PART1: &str = "qwen_coder_part1.safetensors";
const MODEL_PATH_PART2: &str = "qwen_coder_part2.safetensors";
const TOKENIZER_PATH: &str = "qwen_tokenizer.json";
const CONFIG_PATH: &str = "qwen_config.json";
const TOKENIZER_CONFIG_PATH: &str = "qwen_tokenizer_config.json";
const VOCAB_PATH: &str = "qwen_vocab.json";
const MERGES_PATH: &str = "qwen_merges.txt";

const MAX_LENGTH: usize = 2048;
const MAX_NEW_TOKENS: usize = 512;

/// Структура для загрузки и объединения файлов модели
struct ModelDownloader {
    client: Client,
    chunk_size: usize,
    max_retries: u32,
}

impl ModelDownloader {
    fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(3600)) // 1 час таймаут
            .build()?;

        Ok(Self {
            client,
            chunk_size: 10 * 1024 * 1024, // 10MB чанки
            max_retries: 5,
        })
    }

    /// Проверяет размер файла на сервере
    fn get_remote_size(&self, url: &str) -> Result<u64> {
        let response = self.client.head(url).send()?;
        let content_length = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .context("Не удалось получить размер файла")?;
        Ok(content_length)
    }

    /// Загружает чанк файла
    fn download_chunk(&self, url: &str, start: u64, end: u64, file: &mut File, pb: &ProgressBar) -> Result<()> {
        let mut retries = 0;
        loop {
            match self.try_download_chunk(url, start, end, file, pb) {
                Ok(_) => break,
                Err(e) => {
                    retries += 1;
                    if retries >= self.max_retries {
                        return Err(e);
                    }
                    println!("\nОшибка загрузки чанка, попытка {}/{}...", retries + 1, self.max_retries);
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
        Ok(())
    }

    fn try_download_chunk(&self, url: &str, start: u64, end: u64, file: &mut File, pb: &ProgressBar) -> Result<()> {
        let range = format!("bytes={}-{}", start, end);
        let mut response = self.client
            .get(url)
            .header(RANGE, &range)
            .send()
            .context("Ошибка запроса чанка")?;

        file.seek(SeekFrom::Start(start))?;
        let mut buffer = vec![0; 8192];
        while let Ok(n) = response.read(&mut buffer) {
            if n == 0 { break; }
            file.write_all(&buffer[..n])?;
            pb.inc(n as u64);
        }
        Ok(())
    }

    /// Проверяет целостность загруженного файла
    fn verify_file(&self, path: &str, expected_size: u64) -> Result<bool> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len() == expected_size)
    }

    /// Загружает файл с поддержкой докачки
    fn download_file(&self, url: &str, path: &str) -> Result<()> {
        println!("Проверка файла: {}", path);
        
        let total_size = self.get_remote_size(url)?;
        let file_exists = Path::new(path).exists();
        let mut current_size = 0;

        if file_exists {
            let metadata = std::fs::metadata(path)?;
            current_size = metadata.len();
            
            if current_size == total_size {
                println!("Файл уже полностью загружен");
                return Ok(());
            } else if current_size > total_size {
                println!("Файл поврежден, начинаем загрузку заново");
                current_size = 0;
            } else {
                println!("Найден частично загруженный файл: {} / {} байт", current_size, total_size);
            }
        }

        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(path)?;

        let pb = ProgressBar::new(total_size);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        pb.set_position(current_size);

        let mut start = current_size;
        while start < total_size {
            let end = std::cmp::min(start + self.chunk_size as u64 - 1, total_size - 1);
            self.download_chunk(url, start, end, &mut file, &pb)?;
            start = end + 1;
        }

        pb.finish_with_message("Загрузка завершена");
        
        // Проверяем целостность
        if !self.verify_file(path, total_size)? {
            return Err(anyhow!("Ошибка проверки целостности файла"));
        }

        Ok(())
    }

    /// Объединяет части модели
    fn merge_model_parts(&self, part1_path: &str, part2_path: &str, merges_path: &str, output_path: &str) -> Result<()> {
        println!("Объединение частей модели...");

        // Читаем файл merges.txt для получения информации об объединении
        let merges_content = std::fs::read_to_string(merges_path)
            .context("Не удалось прочитать файл merges.txt")?; // TODO: подумать над объединением частей модели

        let mut output_file = File::create(output_path)?;
        
        // Копируем первую часть
        let mut part1 = File::open(part1_path)?;
        std::io::copy(&mut part1, &mut output_file)?;

        // Копируем вторую часть
        let mut part2 = File::open(part2_path)?;
        std::io::copy(&mut part2, &mut output_file)?;

        println!("Модель успешно объединена");
        Ok(())
    }
}

// /// Загружает файл из интернета если он не существует локально
// fn download_if_needed(url: &str, path: &str) -> Result<()> {
//     if !Path::new(path).exists() {
//         println!("Загрузка {}...", path);
//         let client = Client::builder()
//             .timeout(std::time::Duration::from_secs(1200))
//             .build()?;
        
//         let mut response = client.get(url)
//             .send()
//             .context("Не удалось получить ответ от сервера")?;
        
//         let total_size = response.content_length().unwrap_or(0);
        
//         let pb = ProgressBar::new(total_size);
//         pb.set_style(ProgressStyle::default_bar()
//             .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
//             .unwrap()
//             .progress_chars("#>-"));

//         let mut file = std::fs::File::create(path)
//             .context("Не удалось создать файл")?;

//         let mut downloaded: u64 = 0;
        
//         loop {
//             let mut buffer = vec![0; 8192];
//             let bytes_read = response.read(&mut buffer)?;
//             if bytes_read == 0 {
//                 break;
//             }
//             file.write_all(&buffer[..bytes_read])?;
//             downloaded += bytes_read as u64;
//             pb.set_position(downloaded);
//         }
            
//         pb.finish_with_message("Загрузка завершена");
//     }
//     Ok(())
// }

/// Загружает файл из интернета если он не существует локально или требует докачки
fn download_if_needed(url: &str, path: &str) -> Result<()> {
    let downloader = ModelDownloader::new()?;
    downloader.download_file(url, path)
}

/// Загружает и объединяет части модели
fn download_and_merge_model(
    part1_url: &str,
    part2_url: &str,
    merges_url: &str,
    part1_path: &str,
    part2_path: &str,
    merges_path: &str,
    output_path: &str,
) -> Result<()> {
    let downloader = ModelDownloader::new()?;

    // Загружаем все необходимые файлы
    downloader.download_file(part1_url, part1_path)?;
    downloader.download_file(part2_url, part2_path)?;
    downloader.download_file(merges_url, merges_path)?;

    // Объединяем части модели
    downloader.merge_model_parts(part1_path, part2_path, merges_path, output_path)?;

    Ok(())
}

/// Структура для работы с Qwen моделью
struct QwenModel {
    tokenizer: Tokenizer,
    embeddings: GpuEmbeddings,
    config: Value,
    vocab: Value,
}

impl QwenModel {
    /// Загружает модель и токенизатор
    fn new() -> Result<Self> {
        // // Загружаем все необходимые файлы
        // download_if_needed(MODEL_URL_PART1, MODEL_PATH_PART1)?;
        // download_if_needed(MODEL_URL_PART2, MODEL_PATH_PART2)?;
        // download_if_needed(TOKENIZER_URL, TOKENIZER_PATH)?;
        // download_if_needed(CONFIG_URL, CONFIG_PATH)?;
        // download_if_needed(TOKENIZER_CONFIG_URL, TOKENIZER_CONFIG_PATH)?;
        // download_if_needed(VOCAB_URL, VOCAB_PATH)?;
        // download_if_needed(MERGES_URL, MERGES_PATH)?;
        
        // // Загружаем конфигурацию
        // let config: Value = serde_json::from_str(&fs::read_to_string(CONFIG_PATH)?)?;
        // let vocab: Value = serde_json::from_str(&fs::read_to_string(VOCAB_PATH)?)?;

        // Загружаем все необходимые файлы
        download_if_needed(TOKENIZER_URL, TOKENIZER_PATH)?;
        download_if_needed(CONFIG_URL, CONFIG_PATH)?;
        download_if_needed(TOKENIZER_CONFIG_URL, TOKENIZER_CONFIG_PATH)?;
        download_if_needed(VOCAB_URL, VOCAB_PATH)?;
        
        // Загружаем и объединяем части модели
        download_and_merge_model(
            MODEL_URL_PART1,
            MODEL_URL_PART2,
            MERGES_URL,
            MODEL_PATH_PART1,
            MODEL_PATH_PART2,
            MERGES_PATH,
            "qwen_model_merged.safetensors",
        )?;

        // Загружаем конфигурацию и словарь
        let config: Value = serde_json::from_str(&fs::read_to_string(CONFIG_PATH)?)?;
        let vocab: Value = serde_json::from_str(&fs::read_to_string(VOCAB_PATH)?)?;
        
        // Загружаем токенизатор с необходимыми файлами
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow::anyhow!("Не удалось загрузить токенизатор: {}", e))?;

        // Загружаем обе части модели
        let model_data_part1 = fs::read(MODEL_PATH_PART1)?;
        let model_data_part2 = fs::read(MODEL_PATH_PART2)?;
        
        let tensors_part1 = SafeTensors::deserialize(&model_data_part1)?;
        let tensors_part2 = SafeTensors::deserialize(&model_data_part2)?;

        println!("Доступные тензоры в первой части модели:");
        for tensor_name in tensors_part1.names() {
            println!("- {}", tensor_name);
        }

        println!("\nДоступные тензоры во второй части модели:");
        for tensor_name in tensors_part2.names() {
            println!("- {}", tensor_name);
        }

        // Получаем эмбеддинги из первой части модели
        let embeddings_tensor = tensors_part1
            .tensor("model.embed_tokens.weight")
            .or_else(|_| tensors_part1.tensor("transformer.wte.weight"))
            .context("Не найдены word embeddings")?;

        let embeddings_data = unsafe {
            std::slice::from_raw_parts(
                embeddings_tensor.data().as_ptr() as *const f32,
                embeddings_tensor.data().len() / std::mem::size_of::<f32>()
            )
        };

        // Получаем размерность эмбеддингов из конфигурации
        let hidden_size = config["hidden_size"]
            .as_u64()
            .context("Не найден параметр hidden_size в конфигурации")? as usize;

        let embeddings = GpuEmbeddings::new(embeddings_data, hidden_size)?;

        Ok(Self {
            tokenizer,
            embeddings,
            config,
            vocab,
        })
    }

    /// Применяет шаблон чата как в Python-версии
    fn apply_chat_template(&self, messages: &[(&str, &str)]) -> String {
        let mut result = String::new();
        
        for (role, content) in messages {
            result.push_str(&format!("<|im_start|>{}\n{}\n<|im_end|>\n", role, content));
        }
        
        result.push_str("<|im_start|>assistant\n");
        result
    }

    /// Генерирует код на основе промпта
    fn generate_code(&self, prompt: &str) -> Result<String> {
    println!("=== Начало генерации кода ===");
    println!("Входной промпт: {}", prompt);
    
        let messages = vec![
            ("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
            ("user", prompt)
        ];
        
        let formatted_prompt = self.apply_chat_template(&messages);
    println!("Форматированный промпт: {}", formatted_prompt);
        
    // Токенизация
    println!("Выполняем токенизацию...");
        let encoding = self.tokenizer.encode(
            formatted_prompt,
            true
        ).map_err(|e| anyhow::anyhow!("Ошибка токенизации: {}", e))?;

        let input_ids = encoding.get_ids();
    println!("Количество входных токенов: {}", input_ids.len());
        
        if input_ids.len() > self.embeddings.max_sequence_length {
            return Err(anyhow::anyhow!(
                "Длина последовательности превышает максимально допустимую: {} > {}", 
                input_ids.len(), 
                self.embeddings.max_sequence_length
            ));
        }

        unsafe {
        println!("Получаем начальные эмбеддинги...");
            let embeddings = self.embeddings.forward(input_ids)?;
        println!("Размер эмбеддингов: {}", embeddings.len());
            
            let output_size = MAX_NEW_TOKENS;
            let mut generated_ids = Vec::with_capacity(output_size);
        println!("Инициализирован буфер для генерации, макс. размер: {}", output_size);
            
        println!("Создаем буфер для логитов на GPU...");
        let vocab_size = self.vocab.as_object().unwrap().len();
            let logits_buffer = cl_create!(clCreateBuffer(
                self.embeddings.context,
                CL_MEM_READ_WRITE,
            output_size * vocab_size * std::mem::size_of::<f32>(),
                ptr::null_mut(),
                &mut 0
            ))?;
        println!("Буфер логитов создан, размер словаря: {}", vocab_size);

            generated_ids.extend_from_slice(input_ids);
        println!("Добавлены начальные токены");
            
        let mut timeout_counter = 0;
        const MAX_TIMEOUT: i32 = 1000; // Максимальное количество итераций
        
        println!("=== Начало цикла генерации ===");
            while generated_ids.len() < output_size {
            timeout_counter += 1;
            if timeout_counter > MAX_TIMEOUT {
                println!("Превышено максимальное время генерации");
                break;
            }

            println!("\nИтерация {}, токенов: {}", timeout_counter, generated_ids.len());
            
            let current_embeddings = self.embeddings.forward(&generated_ids)?;
            
            println!("Копируем эмбеддинги на GPU...");
                cl_check!(clEnqueueWriteBuffer(
                    self.embeddings.command_queue,
                    self.embeddings.input_buffer,
                    true,
                    0,
                    current_embeddings.len() * std::mem::size_of::<f32>(),
                    current_embeddings.as_ptr() as *const std::ffi::c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut()
                ))?;
                
            println!("Устанавливаем аргументы ядра...");
                cl_check!(clSetKernelArg(
                    self.embeddings.kernel,
                    0,
                    std::mem::size_of::<cl_mem>(),
                    &self.embeddings.input_buffer as *const _ as *const std::ffi::c_void
                ))?;
                
                cl_check!(clSetKernelArg(
                    self.embeddings.kernel,
                    1,
                    std::mem::size_of::<cl_mem>(),
                    &logits_buffer as *const _ as *const std::ffi::c_void
                ))?;

            println!("Запускаем ядро...");
                let global_work_size = generated_ids.len();
                cl_check!(clEnqueueNDRangeKernel(
                    self.embeddings.command_queue,
                    self.embeddings.kernel,
                    1,
                    ptr::null(),
                    &global_work_size,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null_mut()
                ))?;

            println!("Читаем результаты...");
            let mut logits = vec![0.0f32; vocab_size];
                cl_check!(clEnqueueReadBuffer(
                    self.embeddings.command_queue,
                    logits_buffer,
                    true,
                    0,
                    logits.len() * std::mem::size_of::<f32>(),
                    logits.as_mut_ptr() as *mut std::ffi::c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut()
                ))?;

                let temperature = 0.7f32;
            println!("Применяем температуру: {}", temperature);
                for logit in logits.iter_mut() {
                    *logit /= temperature;
                }

                let next_token = logits.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap();
            println!("Выбран следующий токен: {}", next_token);

            let end_token = self.tokenizer.token_to_id("<|im_end|>").unwrap_or(0);
            if next_token == end_token {
                println!("Достигнут токен окончания");
                    break;
                }

                generated_ids.push(next_token);
            }

        println!("Освобождаем ресурсы GPU...");
            cl_check!(clReleaseMemObject(logits_buffer))?;

        println!("Декодируем результат...");
            let generated_text = self.tokenizer.decode(&generated_ids, true)
                .map_err(|e| anyhow::anyhow!("Ошибка детокенизации: {}", e))?;

        println!("Извлекаем код из результата...");
            let code = if let Some(start) = generated_text.find("<|im_start|>assistant\n") {
                let code_start = start + "<|im_start|>assistant\n".len();
                if let Some(end) = generated_text[code_start..].find("<|im_end|>") {
                    generated_text[code_start..code_start + end].trim().to_string()
                } else {
                    generated_text[code_start..].trim().to_string()
                }
            } else {
                generated_text
            };

        println!("=== Генерация завершена ===");
            Ok(code)
        }
    }
}

fn main() -> Result<()> {
    println!("Инициализация Qwen Coder модели...");
    let model = QwenModel::new()?;

    let test_prompts = [
        "Write a quick sort algorithm in Rust",
//        "Create a function to check if a number is prime",
//        "Implement a binary tree structure with basic operations",
    ];

    println!("\nГенерация кода для тестовых промптов:");
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\nПромпт {}: {}", i + 1, prompt);
        
        let generated_code = model.generate_code(prompt)?;
        println!("\nСгенерированный код:\n{}", generated_code);
    }

    Ok(())
}
