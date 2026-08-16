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
- **Positive (2):** 15,382 ulasan (85.82%)
- **Neutral (1):** 886 ulasan (4.94%)
- **Negative (0):** 1,655 ulasan (9.23%)

## 4. Evaluasi Model SVM
Kinerja pengklasifikasi teks Linear SVM pada Test Set (20% split):
- **Akurasi Model:** 88.79%
- **Macro F1-Score:** 0.6125
- *Catatan:* Perincian presisi, recall, dan confusion matrix tersimpan di berkas biner/gambar laporan.

## 5. Analisis Sentimen per Destinasi
Daftar destinasi dengan akumulasi sentimen ulasan (menampilkan destinasi pariwisata terpopuler):

| Nama Destinasi | Total Ulasan | Positif (%) | Netral (%) | Negatif (%) | Avg Rating |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Kampung Muara Sunda | 402 | 66.42% | 3.98% | 29.6% | 4.2 |
| Ramenkane Cikuray Garut | 398 | 98.74% | 0.5% | 0.75% | 4.9 |
| RM Saung Cikenceh (Garut Kota, Cikuray) | 383 | 93.99% | 1.31% | 4.7% | 4.7 |
| Bumi Upi | 375 | 82.67% | 7.47% | 9.87% | 4.5 |
| Joglo Abah Resto, Kedai Kopi dan Pusat Oleh-oleh | 370 | 96.76% | 0.81% | 2.43% | 4.9 |
| Garland Barnville | 350 | 83.14% | 4.0% | 12.86% | 4.4 |
| Botram Garut | 349 | 87.39% | 4.87% | 7.74% | 4.6 |
| Glamping Villa by Sabda Alam | 333 | 96.4% | 1.8% | 1.8% | 4.9 |
| Rumah Makan Lumbung Padi Garut | 311 | 82.64% | 4.18% | 13.18% | 4.5 |
| Ramen Gorilla | 311 | 93.89% | 1.61% | 4.5% | 4.4 |
| Rumah Makan Sunda - Dapoer Nyunda | 310 | 97.74% | 0.65% | 1.61% | 5.0 |
| Kebun Mawar SITUHAPA Samarang | 304 | 90.46% | 3.62% | 5.92% | 4.5 |
| Gunung Papandayan Garut | 304 | 93.42% | 1.97% | 4.61% | 4.7 |
| Rumah Makan Khas Sunda M. Iki | 296 | 82.43% | 8.78% | 8.78% | 4.5 |
| RM Sugema Raya | 293 | 81.91% | 8.87% | 9.22% | 4.5 |

## 6. Analisis Sentimen Berdasarkan Kategori
Agregasi distribusi sentimen berdasarkan jenis/kategori destinasi wisata di Kabupaten Garut:

| Kategori Wisata | Jumlah Destinasi | Total Ulasan | Positif (%) | Netral (%) | Negatif (%) | Avg Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Area Mendaki | 2 | 26 | 92.31% | 3.85% | 3.85% | 4.65 |
| Area Rekreasi Alam | 2 | 2 | 50.0% | 50.0% | 0.0% | 4.85 |
| Bangunan Bersejarah | 1 | 1 | 100.0% | 0.0% | 0.0% | 4.6 |
| Cagar Alam | 2 | 8 | 100.0% | 0.0% | 0.0% | 4.6 |
| Danau | 1 | 3 | 66.67% | 0.0% | 33.33% | 4.4 |
| Danau, Tujuan Wisata | 1 | 110 | 66.36% | 15.45% | 18.18% | 4.25 |
| Hotel | 4 | 114 | 86.84% | 6.14% | 7.02% | 3.2 |
| Hotel Resor | 1 | 3 | 33.33% | 66.67% | 0.0% | 4.5 |
| Kafe | 3 | 299 | 80.94% | 6.35% | 12.71% | 4.57 |
| Kebun Binatang | 1 | 169 | 88.17% | 4.14% | 7.69% | 4.3 |
| Kedai Sarapan & Makan Siang | 1 | 264 | 80.3% | 8.71% | 10.98% | 4.3 |
| Kolam Renang | 8 | 499 | 82.36% | 5.01% | 12.63% | 4.4 |
| Kolam Renang Umum | 2 | 138 | 81.16% | 7.97% | 10.87% | 4.5 |
| Kolam renang luar ruangan | 1 | 24 | 58.33% | 4.17% | 37.5% | 4.3 |
| Kompleks Kolam Renang | 1 | 194 | 74.74% | 7.73% | 17.53% | 4.2 |
| Layanan Sewa Tenda | 1 | 3 | 33.33% | 33.33% | 33.33% | 4.2 |
| Masjid | 1 | 137 | 73.72% | 11.68% | 14.6% | 4.7 |
| Otoritas Pelabuhan | 1 | 2 | 50.0% | 50.0% | 0.0% | 4.8 |
| Pantai | 15 | 1,166 | 81.22% | 5.75% | 13.04% | 4.37 |
| Pantai, Tujuan Wisata | 1 | 14 | 85.71% | 7.14% | 7.14% | 4.5 |
| Pasar Malam | 1 | 75 | 85.33% | 5.33% | 9.33% | 4.5 |
| Pemandian Umum | 1 | 3 | 100.0% | 0.0% | 0.0% | 4.9 |
| Pemandian air panas | 1 | 16 | 93.75% | 6.25% | 0.0% | 4.8 |
| Pemandian di Ruang Terbuka | 3 | 11 | 100.0% | 0.0% | 0.0% | 4.3 |
| Pondok | 2 | 8 | 100.0% | 0.0% | 0.0% | 4.35 |
| Produsen makanan | 1 | 11 | 81.82% | 0.0% | 18.18% | 4.4 |
| Pujasera | 2 | 19 | 73.68% | 15.79% | 10.53% | 4.5 |
| Pusat Informasi Pariwisata | 1 | 4 | 100.0% | 0.0% | 0.0% | 5.0 |
| Restoran | 15 | 2,283 | 88.61% | 2.85% | 8.54% | 4.61 |
| Restoran Bakso | 3 | 90 | 82.22% | 5.56% | 12.22% | 4.57 |
| Restoran Indonesia | 3 | 468 | 82.05% | 7.26% | 10.68% | 4.57 |
| Restoran Jepang | 1 | 311 | 93.89% | 1.61% | 4.5% | 4.4 |
| Restoran Korea | 2 | 358 | 89.39% | 1.68% | 8.94% | 4.6 |
| Restoran Masakan Ayam | 1 | 26 | 88.46% | 3.85% | 7.69% | 4.3 |
| Restoran Mie | 1 | 398 | 98.74% | 0.5% | 0.75% | 4.9 |
| Restoran Sate | 2 | 102 | 86.27% | 2.94% | 10.78% | 4.5 |
| Restoran Seafood | 1 | 31 | 41.94% | 3.23% | 54.84% | 4.2 |
| Restoran Steak | 1 | 112 | 80.36% | 2.68% | 16.96% | 4.4 |
| Restoran Sunda | 12 | 2,139 | 80.32% | 5.19% | 14.49% | 4.41 |
| Restoran makanan pedesaan | 1 | 19 | 68.42% | 0.0% | 31.58% | 4.1 |
| Rumah Makan | 1 | 349 | 87.39% | 4.87% | 7.74% | 4.6 |
| Rumah Pondokan | 1 | 1 | 100.0% | 0.0% | 0.0% | 4.3 |
| Spa | 1 | 18 | 83.33% | 16.67% | 0.0% | 4.4 |
| Taman | 7 | 217 | 88.48% | 5.53% | 5.99% | 4.36 |
| Taman Ekologi | 1 | 35 | 71.43% | 11.43% | 17.14% | 4.2 |
| Taman Hiburan | 2 | 14 | 92.86% | 7.14% | 0.0% | 4.55 |
| Taman Kota | 6 | 289 | 91.35% | 5.88% | 2.77% | 4.6 |
| Taman Rekreasi Air | 1 | 1 | 100.0% | 0.0% | 0.0% | 5.0 |
| Taman bermain | 5 | 43 | 88.37% | 4.65% | 6.98% | 4.76 |
| Tempat Acara Memancing | 1 | 10 | 90.0% | 0.0% | 10.0% | 4.9 |
| Titik Pemandangan | 2 | 7 | 100.0% | 0.0% | 0.0% | 4.85 |
| Toko Makanan | 1 | 229 | 89.52% | 3.49% | 6.99% | 4.5 |
| Tujuan Wisata | 129 | 5,289 | 86.9% | 5.39% | 7.71% | 4.5 |
| Wilayah Lintas Alam | 1 | 5 | 100.0% | 0.0% | 0.0% | 4.4 |

## 7. Promotional Targets
Destinasi pariwisata unggulan dengan reputasi kepuasan publik tinggi (sentimen positif dominan) yang direkomendasikan untuk promosi masif:

| Nama Destinasi | Total Ulasan | Positif (%) | Avg Rating |
| :--- | :---: | :---: | :---: |
| Curug Ciarjuna | 11 | 100.0% | 4.8 |
| Warung Asakan | 10 | 100.0% | 4.7 |
| Curug Citiis | 10 | 100.0% | 4.6 |
| Jip Wisata Garut | 16 | 100.0% | 5.0 |
| Pantai cicalengka | 15 | 100.0% | 4.6 |
| Tujuh Curug Cimanganten | 16 | 100.0% | 4.5 |
| Bukit Dinar Dirham | 16 | 100.0% | 4.8 |
| Pantai Sancang | 34 | 100.0% | 4.7 |
| Batu Tumpang Kab.Garut | 39 | 100.0% | 4.5 |
| Wisata GUNUNG WAYANG | 16 | 100.0% | 4.5 |

## 8. Monitoring / Improvement Targets
Destinasi pariwisata dengan persentase ulasan netral yang relatif tinggi atau belum menunjukkan dominasi persepsi yang kuat:

| Nama Destinasi | Total Ulasan | Netral (%) | Avg Rating |
| :--- | :---: | :---: | :---: |
| Wisata Situ Bagendit 2 | 21 | 23.81% | 4.4 |
| Teras Cimanuk | 18 | 22.22% | 4.7 |
| Taman Kuliner CIbatu | 14 | 21.43% | 4.4 |
| Wisata | 58 | 5.17% | 0.0 |

## 9. Policy Intervention Targets
Destinasi pariwisata yang menunjukkan proporsi ulasan negatif relatif tinggi, direkomendasikan untuk ditinjau langsung oleh dinas terkait:

| Nama Destinasi | Total Ulasan | Negatif (%) | Avg Rating |
| :--- | :---: | :---: | :---: |
| Rumah Makan Kencana Sunda | 47 | 55.32% | 4.0 |
| Warung Bambu garut | 31 | 54.84% | 4.2 |
| Waterboom Tirta Kencana | 33 | 39.39% | 4.0 |
| WISATA BINAR ALAM VIEW | 24 | 37.5% | 4.3 |
| Kedai Itikurih | 132 | 33.33% | 4.3 |
| Rumah Makan Megawati | 12 | 33.33% | 4.4 |
| Kolam renang cipanas indah | 25 | 32.0% | 4.0 |
| Sentral kuliner ikan garut | 19 | 31.58% | 4.1 |
| Kampung Muara Sunda | 402 | 29.6% | 4.2 |
| Sabda Alam Water Park | 11 | 27.27% | 4.4 |

## 10. Interpretasi dan Rekomendasi Kebijakan
Analisis interpretasi ini didasarkan pada data persepsi ulasan ulasan digital pariwisata:

### Rekomendasi Prioritas Intervensi:
- Destinasi **Rumah Makan Kencana Sunda** memiliki proporsi prediksi sentimen negatif sebesar **55.32%** dari total **47** ulasan valid yang dianalisis. Temuan ini menunjukkan adanya ketidakpuasan pengunjung yang cukup tinggi secara statistik, sehingga destinasi tersebut direkomendasikan untuk diprioritaskan dalam evaluasi lapangan lebih lanjut oleh pemangku kepentingan pariwisata Kabupaten Garut.
- Destinasi **Warung Bambu garut** memiliki proporsi prediksi sentimen negatif sebesar **54.84%** dari total **31** ulasan valid yang dianalisis. Temuan ini menunjukkan adanya ketidakpuasan pengunjung yang cukup tinggi secara statistik, sehingga destinasi tersebut direkomendasikan untuk diprioritaskan dalam evaluasi lapangan lebih lanjut oleh pemangku kepentingan pariwisata Kabupaten Garut.
- Destinasi **Waterboom Tirta Kencana** memiliki proporsi prediksi sentimen negatif sebesar **39.39%** dari total **33** ulasan valid yang dianalisis. Temuan ini menunjukkan adanya ketidakpuasan pengunjung yang cukup tinggi secara statistik, sehingga destinasi tersebut direkomendasikan untuk diprioritaskan dalam evaluasi lapangan lebih lanjut oleh pemangku kepentingan pariwisata Kabupaten Garut.

### Analisis Pemantauan (Sentimen Netral):
- Destinasi **Wisata Situ Bagendit 2** menunjukkan proporsi sentimen netral sebesar **23.81%** dari total **21** ulasan valid. Hal ini mengindikasikan bahwa impresi atau persepsi pengunjung terhadap destinasi pariwisata tersebut belum terbentuk ke arah positif maupun negatif secara dominan, sehingga direkomendasikan untuk pemantauan berkelanjutan terkait peningkatan mutu layanan.
- Destinasi **Teras Cimanuk** menunjukkan proporsi sentimen netral sebesar **22.22%** dari total **18** ulasan valid. Hal ini mengindikasikan bahwa impresi atau persepsi pengunjung terhadap destinasi pariwisata tersebut belum terbentuk ke arah positif maupun negatif secara dominan, sehingga direkomendasikan untuk pemantauan berkelanjutan terkait peningkatan mutu layanan.
- Destinasi **Taman Kuliner CIbatu** menunjukkan proporsi sentimen netral sebesar **21.43%** dari total **14** ulasan valid. Hal ini mengindikasikan bahwa impresi atau persepsi pengunjung terhadap destinasi pariwisata tersebut belum terbentuk ke arah positif maupun negatif secara dominan, sehingga direkomendasikan untuk pemantauan berkelanjutan terkait peningkatan mutu layanan.

## 11. Kesimpulan
Sistem analisis sentimen berbasis Linear SVM dan pemetaan kebijakan prioritas ini menyediakan sarana pendukung keputusan (*decision-support tool*) objektif bagi Dinas Pariwisata Kabupaten Garut untuk merencanakan alokasi promosi dan program peningkatan mutu destinasi wisata secara transparan berbasis data (*evidence-based policy*).
