use anyhow::{Context, Result};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use std::fs;
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::io::{Read, Write};
use tch::{Tensor, Device, Kind};
use prettytable::{Table, row};
use ndarray::Array1;
use opencl_neural::{neural::GpuEmbeddings, cl_check, cl_create};

const MODEL_URL: &str = "https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/model.safetensors";
const TOKENIZER_URL: &str = "https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/tokenizer.json";
const CONFIG_URL: &str = "https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/config.json";
const MODEL_PATH: &str = "modernbert_model.safetensors";
const TOKENIZER_PATH: &str = "modernbert_tokenizer.json";
const CONFIG_PATH: &str = "modernbert_config.json";

// Existing download_if_needed function remains the same
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
    // Keep existing new() implementation
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

    // Add a new method specifically for semantic comparison
    // async fn compute_embeddings(&self, texts: &[String]) -> Result<Vec<Array1<f32>>> {
    //     let (hidden_states, attention_mask) = self.text_to_vector(&texts.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
    //     let pooled = self.average_pooling(hidden_states, attention_mask);
        
    //     // Convert tensor to Vec<Array1<f32>>
    //     let data = pooled.to_vec2()?;
    //     Ok(data.into_iter()
    //         .map(|v| Array1::from_vec(v))
    //         .collect())
    // }

    // async fn compute_embeddings(&self, texts: &[String]) -> Result<Vec<Array1<f32>>> {
    //     let (hidden_states, attention_mask) = self.text_to_vector(&texts.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
    //     let pooled = self.average_pooling(hidden_states, attention_mask);
        
    //     // Преобразуем тензор в Vec<f32>
    //     let data: Vec<f32> = pooled.to_kind(Kind::Float).into();  // Убедимся, что тензор имеет тип f32
        
    //     let batch_size = texts.len();
    //     let hidden_size = data.len() / batch_size;
        
    //     // Разбиваем данные на батчи и создаем Array1 для каждого
    //     Ok((0..batch_size)
    //         .map(|i| {
    //             let start = i * hidden_size;
    //             let end = start + hidden_size;
    //             Array1::from_vec(data[start..end].to_vec())
    //         })
    //         .collect())
    // }

    async fn compute_embeddings(&self, texts: &[String]) -> Result<Vec<Array1<f32>>> {
        let (hidden_states, attention_mask) = self.text_to_vector(&texts.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
        let pooled = self.average_pooling(hidden_states, attention_mask);
    
        // Получаем размерности тензора
        let batch_size = texts.len();
        let hidden_size = pooled.size()[1] as usize;
    
        // Преобразуем тензор в массив ndarray
        // let pooled_data = pooled.to_kind(tch::Kind::Float).to_array2::<f32>()?;
        let pooled_data = Vec::<_>::try_from(pooled.to_kind(tch::Kind::Float))?;
    
        // Преобразуем в Vec<Array1<f32>>
        let embeddings = pooled_data
            .into_iter()
            .map(|row| Array1::from_vec(row))
            .collect();
    
        Ok(embeddings)
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

// Helper functions for semantic comparison
fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    ((a - b).mapv(|x| x.powi(2)).sum()).sqrt()
}

fn calculate_statistics(distances: &[f32]) -> (f64, f64, f64) {
    let mut sorted_distances = distances.to_vec();
    sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if sorted_distances.len() % 2 == 0 {
        (sorted_distances[sorted_distances.len() / 2 - 1] as f64 + 
         sorted_distances[sorted_distances.len() / 2] as f64) / 2.0
    } else {
        sorted_distances[sorted_distances.len() / 2] as f64
    };

    let mean = sorted_distances.iter().map(|&x| x as f64).sum::<f64>() / 
               sorted_distances.len() as f64;

    use std::collections::HashMap;
    let mut frequency_map = HashMap::new();
    for &distance in &sorted_distances {
        *frequency_map.entry(distance.to_bits()).or_insert(0) += 1;
    }
    let mode = frequency_map
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(bits, _)| f32::from_bits(bits) as f64)
        .unwrap_or(0.0);

    (mode, median, mean)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Инициализация ModernBERT модели с GPU-ускорением...");
    let model = ModernBertModel::new()?;

    // Списки текстов из разных семантик
    let semantic_list_1 = vec![
        "Собака играет в парке.".to_string(),
        "Щенок гоняется за мячом.".to_string(),
        "Собака лает на прохожих.".to_string(),
        "Собака виляет хвостом.".to_string(),
        "Собака копает яму в саду.".to_string(),
        "Собака охраняет дом.".to_string(),
        "Собака прыгает через препятствие.".to_string(),
        "Собака плавает в озере.".to_string(),
        "Собака спит на коврике.".to_string(),
        "Собака ест кость.".to_string(),
    ];

    let semantic_list_2 = vec![
        "Кот спит на диване.".to_string(),
        "Котенок играет с клубком ниток.".to_string(),
        "Кот мурлычет на солнце.".to_string(),
        "Кот ловит мышь.".to_string(),
        "Кот лежит на подоконнике.".to_string(),
        "Кот пьет молоко.".to_string(),
        "Кот царапает мебель.".to_string(),
        "Кот прыгает на стол.".to_string(),
        "Кот прячется под кроватью.".to_string(),
        "Кот играет с игрушечной мышкой.".to_string(),
    ];

    let test_phrase_1 = "Собака радостно бегает по траве.".to_string();
    let test_phrase_2 = "Кот нежится на подоконнике.".to_string();

    // Compute embeddings using GPU
    let all_texts = [&semantic_list_1[..], &semantic_list_2[..], 
                    &[test_phrase_1.clone(), test_phrase_2.clone()]].concat();
    
    println!("Вычисление эмбеддингов на GPU...");
    let embeddings = model.compute_embeddings(&all_texts).await?;

    // Split embeddings
    let semantic_embeddings_1 = &embeddings[0..semantic_list_1.len()];
    let semantic_embeddings_2 = &embeddings[semantic_list_1.len()..semantic_list_1.len() + semantic_list_2.len()];
    let test_embedding_1 = &embeddings[semantic_list_1.len() + semantic_list_2.len()];
    let test_embedding_2 = &embeddings[semantic_list_1.len() + semantic_list_2.len() + 1];

    // Функция для вычисления евклидова расстояния между двумя векторами
    fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        ((a - b).mapv(|x| x.powi(2)).sum()).sqrt()
    }

    // Функция для извлечения вектора из EmbeddingResult
    // fn extract_vector(embedding_result: &EmbeddingResult) -> Array1<f32> {
    //     match embedding_result {
    //         EmbeddingResult::DenseVector(data) => Array1::from_vec(data.clone()),
    //         EmbeddingResult::MultiVector(data) => {
    //             // Если MultiVector, выбираем первый вектор (или объединяем их)
    //             Array1::from_vec(data[0].clone())
    //         }
    //     }
    // }

    // Преобразуем векторы в ndarray::Array1
    // let test_embedding_1 = extract_vector(test_embedding_1);
    // let test_embedding_2 = extract_vector(test_embedding_2);
    let test_embedding_1 = &embeddings[semantic_list_1.len() + semantic_list_2.len()];
    let test_embedding_2 = &embeddings[semantic_list_1.len() + semantic_list_2.len() + 1];

    // Вычисляем расстояния для test_phrase_1
    let mut distances_1 = Vec::new();
    for (i, embedding) in semantic_embeddings_1.iter().enumerate() {
        distances_1.push((i, "semantic_list_1", euclidean_distance(test_embedding_1, embedding)));
    }
    for (i, embedding) in semantic_embeddings_2.iter().enumerate() {
        distances_1.push((i, "semantic_list_2", euclidean_distance(test_embedding_1, embedding)));
    }

    // Вычисляем расстояния для test_phrase_2
    let mut distances_2 = Vec::new();
    for (i, embedding) in semantic_embeddings_1.iter().enumerate() {
        distances_2.push((i, "semantic_list_1", euclidean_distance(test_embedding_2, embedding)));
    }
    for (i, embedding) in semantic_embeddings_2.iter().enumerate() {
        distances_2.push((i, "semantic_list_2", euclidean_distance(test_embedding_2, embedding)));
    }

    // Сортируем расстояния для test_phrase_1
    distances_1.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Сортируем расстояния для test_phrase_2
    distances_2.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Выводим результаты для test_phrase_1
    let mut table_1 = Table::new();
    table_1.add_row(row!["Проверочная фраза:", test_phrase_1]);
    table_1.add_row(row!["Текст", "Список", "Расстояние"]);

    for (i, list, distance) in distances_1.iter() {
        let text = if *list == "semantic_list_1" {
            &semantic_list_1[*i]
        } else {
            &semantic_list_2[*i]
        };
        table_1.add_row(row![text, list, distance]);
    }

    println!("Результаты для test_phrase_1:");
    table_1.printstd();

    // Выводим результаты для test_phrase_2
    let mut table_2 = Table::new();
    table_2.add_row(row!["Проверочная фраза:", test_phrase_2]);
    table_2.add_row(row!["Текст", "Список", "Расстояние"]);

    for (i, list, distance) in distances_2.iter() {
        let text = if *list == "semantic_list_1" {
            &semantic_list_1[*i]
        } else {
            &semantic_list_2[*i]
        };
        table_2.add_row(row![text, list, distance]);
    }

    println!("\nРезультаты для test_phrase_2:");
    table_2.printstd();

    // Функция для расчёта моды, медианы и среднего значения
    fn calculate_statistics(distances: &[f32]) -> (f64, f64, f64) {
        let mut sorted_distances = distances.to_vec();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Медиана
        let median = if sorted_distances.len() % 2 == 0 {
            (sorted_distances[sorted_distances.len() / 2 - 1] as f64 + sorted_distances[sorted_distances.len() / 2] as f64) / 2.0
        } else {
            sorted_distances[sorted_distances.len() / 2] as f64
        };

        // Среднее значение
        let mean = sorted_distances.iter().map(|&x| x as f64).sum::<f64>() / sorted_distances.len() as f64;

        // Мода
        use std::collections::HashMap;
        let mut frequency_map = HashMap::new();
        for &distance in &sorted_distances {
            *frequency_map.entry(distance.to_bits()).or_insert(0) += 1;
        }
        let mode = frequency_map
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(bits, _)| f32::from_bits(bits) as f64)
            .unwrap_or(0.0);

        (mode, median, mean)
    }

    // Расчёт статистики для test_phrase_1 и semantic_list_1
    let distances_1_list_1: Vec<f32> = distances_1.iter()
        .filter(|(_, list, _)| *list == "semantic_list_1")
        .map(|(_, _, distance)| *distance)
        .collect();
    let (mode_1_list_1, median_1_list_1, mean_1_list_1) = calculate_statistics(&distances_1_list_1);

    // Расчёт статистики для test_phrase_1 и semantic_list_2
    let distances_1_list_2: Vec<f32> = distances_1.iter()
        .filter(|(_, list, _)| *list == "semantic_list_2")
        .map(|(_, _, distance)| *distance)
        .collect();
    let (mode_1_list_2, median_1_list_2, mean_1_list_2) = calculate_statistics(&distances_1_list_2);

    // Расчёт статистики для test_phrase_2 и semantic_list_1
    let distances_2_list_1: Vec<f32> = distances_2.iter()
        .filter(|(_, list, _)| *list == "semantic_list_1")
        .map(|(_, _, distance)| *distance)
        .collect();
    let (mode_2_list_1, median_2_list_1, mean_2_list_1) = calculate_statistics(&distances_2_list_1);

    // Расчёт статистики для test_phrase_2 и semantic_list_2
    let distances_2_list_2: Vec<f32> = distances_2.iter()
        .filter(|(_, list, _)| *list == "semantic_list_2")
        .map(|(_, _, distance)| *distance)
        .collect();
    let (mode_2_list_2, median_2_list_2, mean_2_list_2) = calculate_statistics(&distances_2_list_2);

    // Выводим статистику в таблицу
    let mut stats_table = Table::new();
    stats_table.add_row(row!["Проверочная фраза", "Список", "Мода", "Медиана", "Среднее"]);
    stats_table.add_row(row!["test_phrase_1", "semantic_list_1", mode_1_list_1, median_1_list_1, mean_1_list_1]);
    stats_table.add_row(row!["test_phrase_1", "semantic_list_2", mode_1_list_2, median_1_list_2, mean_1_list_2]);
    stats_table.add_row(row!["test_phrase_2", "semantic_list_1", mode_2_list_1, median_2_list_1, mean_2_list_1]);
    stats_table.add_row(row!["test_phrase_2", "semantic_list_2", mode_2_list_2, median_2_list_2, mean_2_list_2]);

    println!("\nСтатистика расстояний:");
    stats_table.printstd();

    Ok(())
}
