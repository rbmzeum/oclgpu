//! Пример загрузки предобученной модели из формата safetensors
use anyhow::{Context, Result};
use reqwest::blocking::Client;
use safetensors::SafeTensors;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Duration;
use std::io::Read;

const MODEL_URL: &str = "https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors";
const MODEL_PATH: &str = "model.safetensors";

fn download_model(url: &str, path: &str) -> Result<()> {
    println!("Загрузка модели из {}", url);
    
    if Path::new(path).exists() {
        println!("Модель уже загружена");
        return Ok(());
    }

    // Создаем клиент с увеличенным таймаутом
    let client = Client::builder()
        .timeout(Duration::from_secs(300)) // 5 минут таймаут
        .build()
        .context("Не удалось создать HTTP клиент")?;

    // Получаем размер файла
    let response = client.head(url)
        .send()
        .context("Не удалось получить информацию о файле")?;
    
    let total_size = response.content_length()
        .context("Не удалось получить размер файла")?;

    // Начинаем загрузку
    let mut response = client.get(url)
        .send()
        .context("Не удалось начать загрузку")?;

    let mut file = File::create(path)
        .with_context(|| format!("Не удалось создать файл {}", path))?;

    let mut downloaded: u64 = 0;
    let mut buffer = [0; 8192];
    let total_mb = total_size as f64 / 1_048_576.0;

    println!("Общий размер: {:.2} MB", total_mb);

    while let Ok(n) = response.read(&mut buffer) {
        if n == 0 { break; }
        file.write_all(&buffer[..n])
            .context("Ошибка при записи в файл")?;
        
        downloaded += n as u64;
        let progress = (downloaded as f64 / total_size as f64) * 100.0;
        let downloaded_mb = downloaded as f64 / 1_048_576.0;
        
        print!("\rЗагружено: {:.2} MB / {:.2} MB ({:.1}%)", 
               downloaded_mb, total_mb, progress);
        std::io::stdout().flush().ok();
    }

    println!("\nМодель успешно загружена и сохранена в {}", path);
    Ok(())
}

fn load_and_inspect_model(path: &str) -> Result<()> {
    println!("\nЗагрузка модели из файла {}...", path);
    
    let data = fs::read(path)
        .with_context(|| format!("Не удалось прочитать файл {}", path))?;
    
    let tensors = SafeTensors::deserialize(&data)
        .context("Не удалось десериализовать tensors")?;
    
    println!("\nСодержимое модели:");
    for name in tensors.names() {
        let tensor = tensors.tensor(name)
            .with_context(|| format!("Не удалось получить тензор {}", name))?;
        
        println!("\nТензор: {}", name);
        println!("  Форма: {:?}", tensor.shape());
        println!("  Тип данных: {:?}", tensor.dtype());
        
        // Выводим первые несколько значений для каждого тензора
        match tensor.dtype() {
            safetensors::Dtype::F32 => {
                let data = tensor.data();
                // Безопасно преобразуем байты в срез f32
                let values = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const f32,
                        data.len() / std::mem::size_of::<f32>()
                    )
                };
                print!("  Первые значения: ");
                for &value in values.iter().take(5) {
                    print!("{:.4} ", value);
                }
                println!("...");
            },
            safetensors::Dtype::F64 => {
                let data = tensor.data();
                let values = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const f64,
                        data.len() / std::mem::size_of::<f64>()
                    )
                };
                print!("  Первые значения: ");
                for &value in values.iter().take(5) {
                    print!("{:.4} ", value);
                }
                println!("...");
            },
            safetensors::Dtype::I64 => {
                let data = tensor.data();
                let values = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const i64,
                        data.len() / std::mem::size_of::<i64>()
                    )
                };
                print!("  Первые значения: ");
                for &value in values.iter().take(5) {
                    print!("{} ", value);
                }
                println!("...");
            },
            _ => println!("  (Пропущено отображение значений для этого типа данных)"),
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("Демонстрация загрузки предобученной модели из формата safetensors\n");

    // Загружаем модель, если её нет
    download_model(MODEL_URL, MODEL_PATH)?;

    // Загружаем и анализируем модель
    load_and_inspect_model(MODEL_PATH)?;

    Ok(())
} 