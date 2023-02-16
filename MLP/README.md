# IF5281 Classifying FashionMNIST with MLP
Pada praktikum kali ini, teman-teman akan mencoba melatih sebuah model *multilayer percepton* yang dapat mengklasifikasi objek visual dari dataset FashionMNIST dengan menggunakan [PyTorch](https://pytorch.org).
Bagi yang belum familiar dengan PyTorch disarankan mempelajari konsep dasarnya dengan mengikuti [tutorial berikut](https://pytorch.org/tutorials/).

## Dependencies
-- *Lewati bagian ini bagi yang sudah siap working environment di workstation masing-masing* --

Jika menggunakan *local environment* pada PC/laptop masing-masing, pastikan *tools* atau *libraries* berikut ini tersedia:
- [Python 3.x.x](https://www.python.org/)
- [PyTorch](https://pytorch.org/) incl. torchvision
- [Jupyter Lab atau Jupyter Notebook](https://jupyter.org/)

Alternatif lainnya, Anda dapat menggunakan komputasi pada *cloud environment* seperti [Google Colab](https://colab.research.google.com/).

## Warming Up
Berikut disediakan *source code* implementasi klasifikasi objek MNIST yang sempat dibahas di kelas: https://github.com/ghif/IF5281/blob/main/src/05-image_classification.ipynb. 
Silakan digunakan sebagai *starter kit* untuk mengimplementasikan model klasifikasi [FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).

## Challenge: Tweaking your classifier to be "good enough"
Tujuan dari tantangan ini adalah sebagai pemanasan untuk lebih familiar dengan PyTorch dalam menghasilkan dan menyetel (*tuning*) model klasifikasi dengan pendekatan Deep Learning. 
Untuk itu, adapun beberapa tugas yang perlu Anda selesaikan.
1. Buatlah *script* kode PyTorch yang mengimplementasikan *training* model MLP (dense layer neural net) dan proses *inference* untuk mengklasifikasi objek FashionMNIST dengan spesifikasi sebagai berikut: (__poin maks: 30__).
    - 2-layer MLP: Linear -> Tanh (H1) -> Linear -> Tanh (H2) -> Linear (Output)
    - Stochastic Gradient Descent (SGD) optimization
    
    
2. Visualisasikan hasil *loss* dan *accuracy* sepanjang *epoch* berjalan dalam bentuk *time-series plot* (__poin maks: 20__).

3. Lakukan perbaikan terhadap mekanisme *training* sehingga mendapatkan akurasi klasifikasi yang lebih baik. Silakan berkreasi terhadap cara atau trik apapun yang Anda lakukan (misal: menambah layer/memperbesar parameter, menggunakan fungsi non-linear/aktivasi lain, menggunakan metode optimisasi lain atau mengganti konfigurasinya, menambah metode regularisasi, dsb), dengan 1 batasan: menggunakan arsitektur MLP/dense layer (__poin maks: 30__).
    - Test accuracy $\geq 88\%$ (30 poin)
    - $82\% \leq \text{test accuracy} < 88\%$ (20 poin)
    - Test accuracy $< 82\%$ (poin: 10)


4. Berikan penjelasan secara kronologis apa yang Anda lakukan pada langkah 3 beserta alasan mengapa Anda melakukan hal-hal tersebut sehingga mampu mencapai performa akurasi yang diinginkan -- penjelasan dapat dituliskan pada jupyter notebook script yang sama, dengan memanfaatkan mode *Markdown* (__poin maksimum: 20__).

### Deliverables
Hasil pekerjaan Anda dikumpulkan dalam format __[NIM]-[NAMALENGKAP]-IF5281-DL1-MLP.zip__ yang berisi:
- 1 *Jupyter notebook script* (ipynb) yang berisi implementasi tantangan 1 (MLP standar) dan 2 (visualisasi).
- 1 *Jupyter notebook script* (ipynb) yang berisi implementasi tantangan 3 (MLP optimum) dan penulisan tantangan 4.

Batas akhir pengumpulan: __Rabu, 22 Februari 2023, 17:00 WIB__.

Selamat belajar!