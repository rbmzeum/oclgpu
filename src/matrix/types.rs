//! Типы матриц и связанные структуры

/// Тип матриц для вычислений
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixType {
    /// Матрицы заполненные 1 и 2
    OnesAndTwos,
    /// Матрицы заполненные 3 и 4 
    ThreesAndFours,
    /// Случайно заполненные матрицы
    Random
} 