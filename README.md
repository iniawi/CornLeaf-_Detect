# 🌽 SiCorn — Klasifikasi Penyakit Daun Jagung

Landing page satu halaman (Streamlit) untuk model `model_mobilenetv2_daun_jagung.keras`.
Navbar (Beranda, Fitur, Cara Kerja, Tentang Kami) melakukan **scroll ke section**,
bukan pindah halaman.

## 1. Cara Menjalankan

```bash
pip install -r requirements.txt
```

Letakkan file model Anda **`model_mobilenetv2_daun_jagung.keras`** di folder yang sama
dengan `app.py`, lalu jalankan:

```bash
streamlit run app.py
```

Jika file model belum ada, aplikasi tetap bisa dibuka (tampilan landing page tetap
berfungsi) — bagian hasil analisis akan menampilkan pesan "Model belum ditemukan".

## 2. Yang Perlu Disesuaikan

Buka `app.py`, lalu cek bagian **KONFIGURASI** di paling atas:

- **`MODEL_PATH`** — ubah jika nama/lokasi file model Anda berbeda.
- **`CLASS_INFO`** — daftar 4 kelas (label, kondisi, rekomendasi penanganan).
  ⚠️ **Urutan list ini harus sama persis** dengan urutan `class_indices` saat Anda
  melatih model (biasanya urutan alfabetis nama folder dataset, misal: `Common_Rust`,
  `Gray_Leaf_Spot`, `Healthy`, `Northern_Leaf_Blight`). Jika hasil prediksi tertukar,
  ini kemungkinan besar penyebabnya — cek `train_generator.class_indices` di notebook
  training Anda dan samakan urutannya di sini.
- **Ukuran input gambar** terdeteksi otomatis dari `model.input_shape`. Jika gagal
  terbaca, akan memakai `FALLBACK_IMG_SIZE = (224, 224)` — ubah jika model Anda
  dilatih dengan ukuran lain.
- **Preprocessing** saat ini menggunakan rescale sederhana (`/255.0`). Jika saat
  training Anda menggunakan `tf.keras.applications.mobilenet_v2.preprocess_input`,
  ganti baris preprocessing di fungsi `predict_image()` agar sesuai.

## 3. Struktur File

```
.
├── app.py                                  # Aplikasi utama
├── requirements.txt
├── model_mobilenetv2_daun_jagung.keras     # ← letakkan model Anda di sini
└── README.md
```

## 4. Catatan Desain

- Satu file Python tunggal, tanpa multipage — semua section ada di `app.py` dan
  ditandai dengan anchor id (`#beranda`, `#fitur`, `#cara-kerja`, `#tentang`,
  `#analisis`) agar navbar bisa scroll-to-section.
- Ilustrasi ladang jagung di hero section digenerate otomatis sebagai SVG (tidak
  memerlukan file gambar eksternal).
- Tidak ada bagian testimoni (sesuai permintaan).
