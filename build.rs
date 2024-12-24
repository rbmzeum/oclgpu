fn main() {
    println!("cargo:rustc-link-search=/usr/lib");     // Путь к библиотекам OpenCL
    println!("cargo:rustc-link-lib=OpenCL");          // Линковка с libOpenCL.so
}
