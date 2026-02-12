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

## JAX vs PyTorch: Keunggulan Komputasi Paralel

Meskipun PyTorch sangat populer, JAX menawarkan pendekatan yang berbeda dan sering kali lebih unggul untuk komputasi paralel:

1.  **Composability (Komposabilitas)**: Di JAX, transformasi seperti `vmap`, `pmap`, dan `jit` dapat disusun seolah-olah menyusun lego. Anda bisa menulis fungsi untuk sampel tunggal, lalu membungkusnya dengan `vmap` untuk *batching*, dan kemudian `pmap` untuk distribusi multi-perangkat secara instan tanpa mengubah logika inti fungsi tersebut.
2.  **Implicit Vectorization (`vmap`)**: PyTorch sering menghasikan kode yang rumit saat menangani *indexing* dan *broadcasting* manual untuk operasi *batch*. JAX menyederhanakan ini dengan `vmap`, yang secara otomatis memvektorisasi fungsi tanpa beban tambahan pada memori atau logika kode.
3.  **XLA Optimizations**: JAX secara asli menggunakan XLA untuk menggabungkan (*fusing*) kernel komputasi. Dalam skenario paralel, hal ini mengurangi latensi komunikasi antar perangkat dan memaksimalkan penggunaan *bandwidth* perangkat keras.
4.  **Functional Programming**: Karena JAX memaksa penggunaan fungsi murni tanpa *side-effects*, pendistribusian beban kerja ke ribuan *core* menjadi lebih aman secara matematis dan mudah diprediksi hasilnya, berbeda dengan paradigma *imperative* yang mungkin memiliki masalah dengan *global state*.

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

*   **Flax**: *Library* fleksibel untuk membangun neural network yang dikembangkan oleh Google. Salah satu API terbarunya, **Flax NNX**, menawarkan pendekatan berbasis objek (*stateful*) yang lebih intuitif.
*   **Haiku**: *Library* berorientasi objek untuk neural network (serupa dengan Sonnet) yang dikembangkan oleh DeepMind.
*   **Optax**: *Library* khusus untuk optimasi dan *gradient processing*.
*   **Equinox**: Pendekatan lain dalam membangun model Deep Learning dengan JAX yang sangat transparan.

### Mengapa Menggunakan Flax NNX & Optax?

Membangun model langsung dengan JAX murni memerlukan pengelolaan parameter secara manual sebagai *Pytrees*. Meskipun memberikan kontrol maksimal, hal ini seringkali repetitif. Di sinilah **Flax NNX** dan **Optax** berperan:

#### 1. Flax NNX (Stateful JAX)
Flax NNX menyederhanakan pengembangan dengan memperkenalkan **`nnx.Module`**. Manfaat utamanya meliputi:
*   **Manajemen Parameter Otomatis**: Anda mendefinisikan lapisan (misal: `nnx.Linear`) sebagai atribut kelas, dan NNX secara otomatis melacak bobot serta biasnya.
*   **Integrasi Pytree**: NNX tetap bekerja selaras dengan sistem Pytree JAX, namun membungkusnya dalam antarmuka berorientasi objek yang familiar (mirip PyTorch).
*   **Kompilasi JIT yang Mudah**: Fungsi pelatihan dapat dibungkus dengan `@nnx.jit`, yang menangani pembaruan *state* model secara transparan.

#### 2. Optax (Optimasi Standar)
Optax adalah standar industri untuk optimasi di JAX. Manfaat rincinya:
*   **Komposabilitas**: Anda dapat menggabungkan berbagai transformasi gradien (seperti *gradient clipping* dan Adam optimizer) menggunakan `optax.chain`.
*   **Kaya Fitur**: Menyediakan implementasi yang sudah teruji untuk SGD, Adam, AdamW, RMSProp, dan teknik penjadwalan *learning rate*.
*   **Pemisahan Logika**: Memisahkan definisi model dari algoritma optimasi, membuat eksperimen jauh lebih cepat dan bersih.

#### 3. Grain (Dataloader Modern)
Grain adalah *library open-source* dari Google yang dirancang khusus untuk *data loading* yang cepat, deterministik, dan dapat diskalakan pada ekosistem JAX/Flax. Di repositori ini, kita menggunakan Grain untuk menangani *preprocessing* dan *batching* gambar (seperti pada MNIST dan CIFAR-10).

Kelebihan utama Grain:
*   **Deterministik**: Memastikan urutan data yang sama setiap kali pelatihan dijalankan, yang sangat penting untuk reproduksibilitas riset.
*   **Stateless**: Grain memisahkan logika transformasi data dari *state* pembacaan, memudahkan pendistribusian beban kerja di multi-GPU/TPU secara efisien.
*   **Pythonic & Flexible**: Memungkinkan penggunaan transformasi standar Python (seperti NumPy dan PIL) langsung di dalam pipeline data.
*   **Integrasi Flax NNX**: Grain bekerja sama sangat baik dengan loop pelatihan NNX, memberikan aliran data yang stabil tanpa menghambat (*bottleneck*) kecepatan komputasi XLA.

**Mengapa tidak menggunakan Dataloader PyTorch?**
Meskipun Dataloader PyTorch bisa digunakan dengan JAX, Grain menawarkan integrasi yang lebih asli (*native*) dengan JAX PRNG (bilangan acak) dan dioptimalkan untuk menghindari *overhead* memori yang sering muncul saat mencampur *framework* yang berbeda dalam satu pipeline.

**Ringkasnya**, beralih dari JAX murni ke Flax NNX & Optax memungkinkan Anda fokus pada arsitektur dan eksperimen, alih-alih terjebak dalam pengelolaan *boilerplate* stateful yang rumit.

## Contoh Sederhana: Deep Learning di JAX

Berikut adalah gambaran bagaimana kita mendefinisikan model menggunakan JAX murni atau bantuan *ecosystem*:

1.  **Definisikan Model**: Menggunakan Flax atau Haiku untuk mendefinisikan *layers*.
2.  **Inisialisasi**: Menginisialisasi parameter model dengan `PRNGKey`.
3.  **Loss Function**: Menghitung *error* model.
4.  **Update Rule**: Menggunakan `grad()` untuk mendapatkan gradien dan Optax untuk memperbarui parameter.

---

JAX menawarkan kontrol yang lebih mendalam bagi Anda yang ingin bereksperimen dengan arsitektur baru atau komputasi yang sangat efisien. Selamat bereksperimen dengan JAX!
