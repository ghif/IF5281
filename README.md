# IF5281 - Materi Praktikum untuk Mata Kuliah Deep Learning, Institut Teknologi Bandung

Repositori ini berisi materi praktikum dan contoh kode untuk mata kuliah **IF5281**. Materi ini awalnya disusun menggunakan framework **PyTorch** (pada tahun akademik 2023 dan 2024), namun saat ini sedang dalam proses optimisasi dan migrasi menggunakan framework **JAX** untuk performa yang lebih tinggi.

## Struktur Repositori

Repositori ini diatur berdasarkan tahun akademik dan topik khusus:

*   **`2023/`**: Materi dan tugas praktikum dari tahun akademik 2023.
*   **`2024/`**: Materi dan tugas praktikum dari tahun akademik 2024.
*   **`jax/`**: Koleksi materi pembelajaran khusus untuk framework JAX, mencakup:
    *   `1-maths/`: Dasar-dasar matematika (vektor, matriks, transformasi).
    *   `2-mlp/`: Implementasi Multi-Layer Perceptron.
    *   `3-convnet/`: Implementasi Convolutional Neural Networks.
    *   `4-rnn/`: Implementasi Recurrent Neural Networks (termasuk text generation).
    *   `5-generative/`: Model-model generatif.

## Materi JAX

Direktori `jax` berisi tutorial interaktif dalam bentuk Jupyter Notebooks. Beberapa topik utama meliputi:

- **Mekanisme PRNG**: Penjelasan mendalam tentang bagaimana JAX menangani bilangan acak secara *stateless* dan deterministik.
- **Transformasi JAX**: Contoh penggunaan `jit`, `grad`, `vmap`, dan `pmap`.
- **Ecosystem**: Pengantar untuk library pendukung seperti Flax, Haiku, dan Optax.

## Persyaratan (Requirements)

Untuk menjalankan kode di repositori ini, disarankan menggunakan environment Python 3.8+ dengan dependensi berikut:

```bash
pip install -r requirements.txt
```

Dependensi utama mencakup `jax`, `jaxlib`, `flax`, dan `optax`.

## Kontribusi

Materi ini dikembangkan untuk kebutuhan internal IF5281, Institut Teknologi Bandung.
