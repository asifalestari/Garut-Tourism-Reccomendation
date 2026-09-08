# Analisis Sentimen Ulasan Destinasi Wisata
## Kabupaten Garut

## 1. Dataset Overview
- **Total Ulasan Valid Setelah Preprocessing:** 17,923 ulasan
- **Total Destinasi Wisata yang Terdaftar:** 272 destinasi
- **Sumber Data:** Google Maps Reviews (Scraped Dataset)

## 2. Distribusi Rating
Distribusi rating ulasan individu (*Individual Review Rating*) dari seluruh dataset:

| Rating Bintang | Jumlah Ulasan | Persentase |
| :--- | :---: | :---: |
| 1.0 Bintang | 1,143 | 6.38% |
| 2.0 Bintang | 408 | 2.28% |
| 3.0 Bintang | 818 | 4.56% |
| 3.5 Bintang | 1 | 0.01% |
| 4.0 Bintang | 2,254 | 12.58% |
| 4.2 Bintang | 1 | 0.01% |
| 4.5 Bintang | 1 | 0.01% |
| 5.0 Bintang | 13,297 | 74.19% |

## 3. Distribusi Sentimen
Distribusi prediksi sentimen keseluruhan ulasan pariwisata:
- **Positive (2):** 16,700 ulasan (93.18%)
- **Neutral (1):** 117 ulasan (0.65%)
- **Negative (0):** 1,106 ulasan (6.17%)

## 4. Evaluasi Model SVM
Kinerja pengklasifikasi teks Linear SVM pada Test Set (20% split):
- **Akurasi Model:** 90.43%
- **Macro F1-Score:** 0.5407
- *Catatan:* Perincian presisi, recall, dan confusion matrix tersimpan di berkas biner/gambar laporan.

## 5. Analisis Sentimen per Destinasi
Daftar destinasi dengan akumulasi sentimen ulasan (menampilkan destinasi pariwisata terpopuler):

| Nama Destinasi | Total Ulasan | Positif (%) | Netral (%) | Negatif (%) | Avg Rating |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Kampung Muara Sunda | 402 | 75.62% | 0.25% | 24.13% | 4.2 |
| Ramenkane Cikuray Garut | 398 | 99.5% | 0.0% | 0.5% | 4.9 |
| RM Saung Cikenceh (Garut Kota, Cikuray) | 383 | 96.34% | 0.26% | 3.39% | 4.7 |
| Bumi Upi | 375 | 92.8% | 1.33% | 5.87% | 4.5 |
| Joglo Abah Resto, Kedai Kopi dan Pusat Oleh-oleh | 370 | 97.57% | 0.27% | 2.16% | 4.9 |
| Garland Barnville | 350 | 91.43% | 0.57% | 8.0% | 4.4 |
| Botram Garut | 349 | 93.12% | 0.57% | 6.3% | 4.6 |
| Glamping Villa by Sabda Alam | 333 | 98.8% | 0.0% | 1.2% | 4.9 |
| Rumah Makan Lumbung Padi Garut | 311 | 91.96% | 0.64% | 7.4% | 4.5 |
| Ramen Gorilla | 311 | 97.11% | 0.32% | 2.57% | 4.4 |
| Rumah Makan Sunda - Dapoer Nyunda | 310 | 99.35% | 0.0% | 0.65% | 5.0 |
| Kebun Mawar SITUHAPA Samarang | 304 | 95.39% | 0.0% | 4.61% | 4.5 |
| Gunung Papandayan Garut | 304 | 97.37% | 0.0% | 2.63% | 4.7 |
| Rumah Makan Khas Sunda M. Iki | 296 | 94.26% | 0.34% | 5.41% | 4.5 |
| RM Sugema Raya | 293 | 92.83% | 0.34% | 6.83% | 4.5 |

## 6. Analisis Sentimen Berdasarkan Kategori
Agregasi distribusi sentimen berdasarkan jenis/kategori destinasi wisata di Kabupaten Garut:

| Kategori Wisata | Jumlah Destinasi | Total Ulasan | Positif (%) | Netral (%) | Negatif (%) | Avg Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Area Mendaki | 2 | 26 | 92.31% | 3.85% | 3.85% | 4.65 |
| Area Rekreasi Alam | 2 | 2 | 100.0% | 0.0% | 0.0% | 4.85 |
| Bangunan Bersejarah | 1 | 1 | 100.0% | 0.0% | 0.0% | 4.6 |
| Cagar Alam | 2 | 8 | 100.0% | 0.0% | 0.0% | 4.6 |
| Danau | 1 | 3 | 66.67% | 0.0% | 33.33% | 4.4 |
| Danau, Tujuan Wisata | 1 | 110 | 73.64% | 10.0% | 16.36% | 4.25 |
| Hotel | 4 | 114 | 94.74% | 0.0% | 5.26% | 3.2 |
| Hotel Resor | 1 | 3 | 100.0% | 0.0% | 0.0% | 4.5 |
| Kafe | 3 | 299 | 90.3% | 1.34% | 8.36% | 4.57 |
| Kebun Binatang | 1 | 169 | 97.04% | 0.0% | 2.96% | 4.3 |
| Kedai Sarapan & Makan Siang | 1 | 264 | 94.7% | 0.38% | 4.92% | 4.3 |
| Kolam Renang | 8 | 499 | 91.58% | 0.0% | 8.42% | 4.4 |
| Kolam Renang Umum | 2 | 138 | 92.03% | 0.72% | 7.25% | 4.5 |
| Kolam renang luar ruangan | 1 | 24 | 87.5% | 0.0% | 12.5% | 4.3 |
| Kompleks Kolam Renang | 1 | 194 | 90.72% | 0.52% | 8.76% | 4.2 |
| Layanan Sewa Tenda | 1 | 3 | 100.0% | 0.0% | 0.0% | 4.2 |
| Masjid | 1 | 137 | 91.97% | 2.19% | 5.84% | 4.7 |
| Otoritas Pelabuhan | 1 | 2 | 100.0% | 0.0% | 0.0% | 4.8 |
| Pantai | 15 | 1,166 | 88.51% | 1.46% | 10.03% | 4.37 |
| Pantai, Tujuan Wisata | 1 | 14 | 100.0% | 0.0% | 0.0% | 4.5 |
| Pasar Malam | 1 | 75 | 90.67% | 0.0% | 9.33% | 4.5 |
| Pemandian Umum | 1 | 3 | 100.0% | 0.0% | 0.0% | 4.9 |
| Pemandian air panas | 1 | 16 | 93.75% | 6.25% | 0.0% | 4.8 |
| Pemandian di Ruang Terbuka | 3 | 11 | 100.0% | 0.0% | 0.0% | 4.3 |
| Pondok | 2 | 8 | 100.0% | 0.0% | 0.0% | 4.35 |
| Produsen makanan | 1 | 11 | 90.91% | 0.0% | 9.09% | 4.4 |
| Pujasera | 2 | 19 | 100.0% | 0.0% | 0.0% | 4.5 |
| Pusat Informasi Pariwisata | 1 | 4 | 100.0% | 0.0% | 0.0% | 5.0 |
| Restoran | 15 | 2,283 | 94.17% | 0.31% | 5.52% | 4.61 |
| Restoran Bakso | 3 | 90 | 91.11% | 1.11% | 7.78% | 4.57 |
| Restoran Indonesia | 3 | 468 | 92.52% | 1.07% | 6.41% | 4.57 |
| Restoran Jepang | 1 | 311 | 97.11% | 0.32% | 2.57% | 4.4 |
| Restoran Korea | 2 | 358 | 94.41% | 0.0% | 5.59% | 4.6 |
| Restoran Masakan Ayam | 1 | 26 | 92.31% | 0.0% | 7.69% | 4.3 |
| Restoran Mie | 1 | 398 | 99.5% | 0.0% | 0.5% | 4.9 |
| Restoran Sate | 2 | 102 | 94.12% | 0.0% | 5.88% | 4.5 |
| Restoran Seafood | 1 | 31 | 58.06% | 0.0% | 41.94% | 4.2 |
| Restoran Steak | 1 | 112 | 88.39% | 0.0% | 11.61% | 4.4 |
| Restoran Sunda | 12 | 2,139 | 88.41% | 0.28% | 11.31% | 4.41 |
| Restoran makanan pedesaan | 1 | 19 | 84.21% | 0.0% | 15.79% | 4.1 |
| Rumah Makan | 1 | 349 | 93.12% | 0.57% | 6.3% | 4.6 |
| Rumah Pondokan | 1 | 1 | 100.0% | 0.0% | 0.0% | 4.3 |
| Spa | 1 | 18 | 100.0% | 0.0% | 0.0% | 4.4 |
| Taman | 7 | 217 | 96.31% | 0.0% | 3.69% | 4.36 |
| Taman Ekologi | 1 | 35 | 82.86% | 0.0% | 17.14% | 4.2 |
| Taman Hiburan | 2 | 14 | 100.0% | 0.0% | 0.0% | 4.55 |
| Taman Kota | 6 | 289 | 98.27% | 0.35% | 1.38% | 4.6 |
| Taman Rekreasi Air | 1 | 1 | 100.0% | 0.0% | 0.0% | 5.0 |
| Taman bermain | 5 | 43 | 93.02% | 0.0% | 6.98% | 4.76 |
| Tempat Acara Memancing | 1 | 10 | 90.0% | 0.0% | 10.0% | 4.9 |
| Titik Pemandangan | 2 | 7 | 100.0% | 0.0% | 0.0% | 4.85 |
| Toko Makanan | 1 | 229 | 96.51% | 0.0% | 3.49% | 4.5 |
| Tujuan Wisata | 129 | 5,289 | 94.27% | 0.96% | 4.76% | 4.5 |
| Wilayah Lintas Alam | 1 | 5 | 100.0% | 0.0% | 0.0% | 4.4 |

## 7. Promotional Targets
Destinasi pariwisata unggulan dengan reputasi kepuasan publik tinggi (sentimen positif dominan) yang direkomendasikan untuk promosi masif:

| Nama Destinasi | Total Ulasan | Positif (%) | Avg Rating |
| :--- | :---: | :---: | :---: |
| Wisata Sungai Ciharus | 41 | 100.0% | 4.5 |
| Curug Rahong | 16 | 100.0% | 4.5 |
| Sumber mata air sirahna | 10 | 100.0% | 4.7 |
| Rumah Makan Dua Saudara | 27 | 100.0% | 4.4 |
| Pabrik Teh Orthodoks Dayeuhmanggung - PTPN | 23 | 100.0% | 4.6 |
| Desa Wisata Situ Cangkuang | 11 | 100.0% | 4.7 |
| Reverdose | 12 | 100.0% | 4.6 |
| Puncak Parabon Kamojang | 18 | 100.0% | 4.4 |
| Jip Wisata Garut | 16 | 100.0% | 5.0 |
| Karacak Valley | 75 | 100.0% | 4.5 |

## 8. Monitoring / Improvement Targets
Destinasi pariwisata dengan persentase ulasan netral yang relatif tinggi atau belum menunjukkan dominasi persepsi yang kuat:

| Nama Destinasi | Total Ulasan | Netral (%) | Avg Rating |
| :--- | :---: | :---: | :---: |
| Wisata | 58 | 0.0% | 0.0 |

## 9. Policy Intervention Targets
Destinasi pariwisata yang menunjukkan proporsi ulasan negatif relatif tinggi, direkomendasikan untuk ditinjau langsung oleh dinas terkait:

| Nama Destinasi | Total Ulasan | Negatif (%) | Avg Rating |
| :--- | :---: | :---: | :---: |
| Rumah Makan Kencana Sunda | 47 | 44.68% | 4.0 |
| Warung Bambu garut | 31 | 41.94% | 4.2 |
| Waterboom Tirta Kencana | 33 | 30.3% | 4.0 |
| Kedai Itikurih | 132 | 28.79% | 4.3 |
| Kampung Muara Sunda | 402 | 24.13% | 4.2 |
| Wisata Pantai Santolo | 90 | 17.78% | 4.3 |
| Kamojang Ecopark | 35 | 17.14% | 4.2 |
| Racik Desa | 88 | 17.05% | 4.3 |
| Rumah Makan Megawati | 12 | 16.67% | 4.4 |
| Pantai santolo | 275 | 16.36% | 4.5 |

## 10. Interpretasi dan Rekomendasi Kebijakan
Analisis interpretasi ini didasarkan pada data persepsi ulasan ulasan digital pariwisata:

### Rekomendasi Prioritas Intervensi:
- Destinasi **Rumah Makan Kencana Sunda** memiliki proporsi prediksi sentimen negatif sebesar **44.68%** dari total **47** ulasan valid yang dianalisis. Temuan ini menunjukkan adanya ketidakpuasan pengunjung yang cukup tinggi secara statistik, sehingga destinasi tersebut direkomendasikan untuk diprioritaskan dalam evaluasi lapangan lebih lanjut oleh pemangku kepentingan pariwisata Kabupaten Garut.
- Destinasi **Warung Bambu garut** memiliki proporsi prediksi sentimen negatif sebesar **41.94%** dari total **31** ulasan valid yang dianalisis. Temuan ini menunjukkan adanya ketidakpuasan pengunjung yang cukup tinggi secara statistik, sehingga destinasi tersebut direkomendasikan untuk diprioritaskan dalam evaluasi lapangan lebih lanjut oleh pemangku kepentingan pariwisata Kabupaten Garut.
- Destinasi **Waterboom Tirta Kencana** memiliki proporsi prediksi sentimen negatif sebesar **30.3%** dari total **33** ulasan valid yang dianalisis. Temuan ini menunjukkan adanya ketidakpuasan pengunjung yang cukup tinggi secara statistik, sehingga destinasi tersebut direkomendasikan untuk diprioritaskan dalam evaluasi lapangan lebih lanjut oleh pemangku kepentingan pariwisata Kabupaten Garut.

### Analisis Pemantauan (Sentimen Netral):
- Destinasi **Wisata** menunjukkan proporsi sentimen netral sebesar **0.0%** dari total **58** ulasan valid. Hal ini mengindikasikan bahwa impresi atau persepsi pengunjung terhadap destinasi pariwisata tersebut belum terbentuk ke arah positif maupun negatif secara dominan, sehingga direkomendasikan untuk pemantauan berkelanjutan terkait peningkatan mutu layanan.

## 11. Kesimpulan
Sistem analisis sentimen berbasis Linear SVM dan pemetaan kebijakan prioritas ini menyediakan sarana pendukung keputusan (*decision-support tool*) objektif bagi Dinas Pariwisata Kabupaten Garut untuk merencanakan alokasi promosi dan program peningkatan mutu destinasi wisata secara transparan berbasis data (*evidence-based policy*).
