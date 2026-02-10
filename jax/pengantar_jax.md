# Pengantar JAX untuk Deep Learning

JAX adalah *framework* komputasi numerik berperforma tinggi yang dikembangkan oleh Google Research. JAX sering disebut sebagai "NumPy yang bisa berjalan di akselerator (GPU/TPU) dan memiliki sistem transformasi fungsi yang kuat".

## Mengapa JAX?

JAX sangat populer di komunitas riset AI karena desainnya yang mengikuti paradigma *functional programming* dan fleksibilitasnya yang luar biasa. Berikut adalah alasan utama mengapa JAX unggul:

1.  **NumPy-like API**: Jika Anda sudah terbiasa dengan NumPy, Anda akan merasa familiar dengan `jax.numpy`.
2.  **Autograd**: JAX dapat secara otomatis menghitung turunan (gradien) dari fungsi Python dan NumPy Anda.
3.  **XLA (Accelerated Linear Algebra)**: JAX menggunakan *compiler* XLA untuk mengoptimalkan operasi aljabar linier, membuatnya sangat cepat di GPU dan TPU melalui kompilasi JIT.

## Transformasi Inti JAX

Kekuatan utama JAX terletak pada empat transformasi fungsi utamanya:

*   **`jit()` (Just-In-Time compilation)**: Mengompilasi fungsi Python Anda menjadi kode mesin yang sangat efisien menggunakan XLA.
*   **`grad()` (Automatic Differentiation)**: Menghasilkan fungsi baru yang menghitung gradien dari fungsi asli.
*   **`vmap()` (Vectorization)**: Secara otomatis memvektorisasi fungsi yang bekerja pada sampel tunggal agar dapat bekerja pada *batch* data dengan efisiensi tinggi.
*   **`pmap()` (Parallelization)**: Mendistribusikan komputasi ke berbagai perangkat (multi-GPU atau multi-TPU) secara paralel.

## Konsep Penting: Stateless PRNG

Berbeda dengan NumPy atau PyTorch yang menggunakan *state* global untuk bilangan acak, JAX mengharuskan penggunaan *key* secara eksplisit:

```python
import jax
key = jax.random.PRNGKey(0)
# Setiap kali ingin angka baru, kita harus split key-nya
key, subkey = jax.random.split(key)
random_data = jax.random.normal(subkey, (10,))
```
Hal ini krusial untuk memastikan reproduksibilitas komputasi paralel dan fungsionalitas murni.

## Ekosistem JAX untuk Deep Learning

JAX sendiri adalah *library* level rendah. Untuk membangun model Deep Learning yang kompleks, komunitas JAX menyediakan berbagai *library* di atasnya:

*   **Flax**: *Library* fleksibel untuk membangun neural network yang dikembangkan oleh Google.
*   **Haiku**: *Library* berorientasi objek untuk neural network (serupa dengan Sonnet) yang dikembangkan oleh DeepMind.
*   **Optax**: *Library* khusus untuk optimasi dan *gradient processing*.
*   **Equinox**: Pendekatan lain dalam membangun model Deep Learning dengan JAX yang sangat transparan.

## Contoh Sederhana: Deep Learning di JAX

Berikut adalah gambaran bagaimana kita mendefinisikan model menggunakan JAX murni atau bantuan *ecosystem*:

1.  **Definisikan Model**: Menggunakan Flax atau Haiku untuk mendefinisikan *layers*.
2.  **Inisialisasi**: Menginisialisasi parameter model dengan `PRNGKey`.
3.  **Loss Function**: Menghitung *error* model.
4.  **Update Rule**: Menggunakan `grad()` untuk mendapatkan gradien dan Optax untuk memperbarui parameter.

---

JAX menawarkan kontrol yang lebih mendalam bagi Anda yang ingin bereksperimen dengan arsitektur baru atau komputasi yang sangat efisien. Selamat bereksperimen dengan JAX!
