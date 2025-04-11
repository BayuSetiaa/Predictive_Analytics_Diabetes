# Laporan Proyek Machine Learning - Bayu Setia

## Domain Proyek

Diabetes mellitus adalah salah satu penyakit kronis yang paling banyak diderita secara global, dan seringkali tidak terdeteksi pada tahap awal. Berdasarkan data dari International Diabetes Federation (IDF), diperkirakan terdapat 537 juta orang dewasa yang hidup dengan diabetes pada tahun 2021, dan jumlah ini diproyeksikan meningkat menjadi 643 juta pada tahun 2030. Sayangnya, lebih dari 50% penderita diabetes tidak terdiagnosis, terutama di negara-negara berkembang yang memiliki keterbatasan fasilitas kesehatan.

**Rubrik/Kriteria Tambahan (Opsional)**:
Masalah ini menjadi sangat krusial karena diabetes yang tidak ditangani sejak dini dapat menyebabkan komplikasi serius seperti gagal ginjal, kebutaan, penyakit jantung, hingga amputasi anggota tubuh. Oleh karena itu, deteksi dini menjadi langkah vital untuk mencegah komplikasi jangka panjang dan menurunkan biaya pengobatan.

Namun, proses diagnosis dini seringkali bergantung pada pemeriksaan medis yang kompleks dan mahal, serta keterbatasan tenaga medis di fasilitas layanan kesehatan primer. Dalam konteks ini, pemanfaatan machine learning untuk prediksi diabetes berdasarkan data kesehatan dasar menjadi solusi yang potensial.

Dengan bantuan model klasifikasi, sistem prediksi dapat dibuat untuk mengidentifikasi kemungkinan seseorang mengidap diabetes dengan hanya menggunakan beberapa indikator sederhana seperti kadar glukosa, BMI, tekanan darah, dan usia. Solusi ini memungkinkan proses skrining otomatis yang lebih cepat dan efisien, terutama di daerah dengan sumber daya terbatas.

Referensi: 1. [Diabetes Prediction Using Ensembling of Different Machine Learning Classifiers](https://ieeexplore.ieee.org/abstract/document/9076634) 2. [Analysis and Prediction of Diabetes Using Machine Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3368308)

## Business Understanding

Diabetes adalah penyakit kronis yang berkembang secara perlahan dan seringkali tidak terdeteksi hingga mencapai tahap serius. Di banyak wilayah, khususnya dengan keterbatasan akses ke layanan kesehatan dan pemeriksaan laboratorium, banyak individu yang tidak mengetahui bahwa mereka menderita diabetes. Ketidaktahuan ini mengakibatkan keterlambatan pengobatan dan meningkatkan risiko komplikasi jangka panjang.

Proses diagnosis diabetes umumnya membutuhkan uji laboratorium dan tenaga medis yang memadai, yang dapat menjadi kendala di fasilitas kesehatan primer, terutama di wilayah terpencil. Maka dibutuhkan solusi yang cepat, efisien, dan terjangkau untuk membantu proses skrining dini diabetes secara otomatis.

### Problem Statements

- Banyak kasus diabetes yang tidak terdiagnosis pada tahap awal, terutama di wilayah yang memiliki keterbatasan fasilitas kesehatan dan tenaga medis. Ketidaktahuan ini menyebabkan pasien datang terlambat ke fasilitas layanan kesehatan dan sudah berada dalam kondisi kronis atau mengalami komplikasi berat.
- Proses diagnosis diabetes umumnya membutuhkan pemeriksaan laboratorium dan analisis manual oleh tenaga medis. Hal ini membutuhkan waktu, biaya, dan sumber daya medis yang tidak selalu tersedia di daerah terpencil atau padat pasien.
- Tenaga medis tidak memiliki alat bantu yang dapat melakukan skrining awal secara otomatis dengan memanfaatkan data kesehatan dasar pasien, padahal data tersebut sering tersedia (seperti usia, kadar glukosa, tekanan darah, dan BMI).

### Goals

- Mengembangkan sistem prediksi diabetes berbasis machine learning yang mampu mendeteksi kemungkinan diabetes secara lebih awal, sehingga potensi komplikasi dapat dikurangi melalui intervensi medis lebih cepat.
- Menyediakan solusi diagnosis alternatif yang murah, cepat, dan tidak membutuhkan pemeriksaan laboratorium kompleks, dengan memanfaatkan data kesehatan yang umum tersedia di fasilitas primer.
- Membuat model prediksi otomatis yang dapat digunakan oleh tenaga medis sebagai alat bantu dalam proses skrining pasien, sehingga proses diagnosis menjadi lebih efisien dan terstandardisasi.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:

  ### Solution statements

  - Solusi 1 – Baseline Model: Logistic Regression Menggunakan Logistic Regression sebagai baseline karena:
    • Cocok untuk klasifikasi biner
    • Mudah diinterpretasikan
    • Cepat dan efisien dalam pelatihan dan prediksi
  - Solusi 2 – Eksperimen dan Peningkatan Model: Random Forest & XGBoost
    • Random Forest dan XGBoost diuji sebagai model alternatif yang lebih kompleks dan mampu menangani hubungan non-linear antar fitur.
    • Dilakukan feature engineering untuk meningkatkan informasi yang diberikan ke model.
    • Mengatasi ketidakseimbangan kelas dengan SMOTE (Synthetic Minority Oversampling Technique).
    • Meningkatkan performa model dengan hyperparameter tuning.
    • Menyesuaikan threshold klasifikasi untuk memaksimalkan recall dan f1-score pada kelas positif (Outcome = 1 / diabetes).

## Data Understanding

Proyek ini menggunakan dataset Pima Indians Diabetes yang merupakan salah satu dataset klasik dalam bidang medis dan machine learning. Dataset ini berisi data kesehatan dasar dari sejumlah perempuan keturunan Pima Indian di Amerika Serikat. Tujuan utama dari dataset ini adalah untuk memprediksi apakah seseorang menderita diabetes berdasarkan beberapa parameter medis.

Dataset ini bersifat terbuka dan banyak digunakan dalam eksperimen klasifikasi biner. Data telah dibersihkan secara umum dan tidak memiliki nilai kosong (missing values), meskipun beberapa nilai seperti 0 pada fitur seperti Glucose, BloodPressure, dan BMI bisa dianggap sebagai anomali atau nilai yang tidak valid.

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:

### Variabel-variabel pada datasetnya adalah sebagai berikut:

Berikut ini adalah penjelasan setiap fitur/variabel dalam dataset:

- Pregnancies: Jumlah kehamilan yang pernah dialami.
- Glucose: Konsentrasi glukosa dalam plasma darah.
- BloodPressure: Tekanan darah diastolik (mm Hg).
- SkinThickness: Ketebalan lipatan kulit triceps (mm).
- Insulin: Konsentrasi insulin serum 2 jam (mu U/ml).
- BMI: Indeks massa tubuh (berat badan dalam kg dibagi kuadrat tinggi badan dalam m²).
- DiabetesPedigreeFunction: Indikator fungsi keturunan terhadap diabetes.
- Age: Usia pasien (dalam tahun).
- Outcome: Label target (0 = tidak diabetes, 1 = positif diabetes).

**Rubrik/Kriteria Tambahan (Opsional)**:

- Distribusi Kelas: Dataset ini bersifat imbalanced, dengan sekitar 65% kelas 0 (negatif) dan 35% kelas 1 (positif).
- Korelasi Fitur: Fitur yang memiliki korelasi paling tinggi terhadap label Outcome adalah Glucose, BMI, dan Age.
- Distribusi Fitur:
  • Beberapa fitur memiliki nilai 0 yang tidak realistis untuk data medis, seperti pada Insulin, SkinThickness, dan BloodPressure, yang kemudian dipertimbangkan untuk ditangani dalam data preparation.
- Visualisasi:
  • Boxplot dan histogram digunakan untuk melihat distribusi dan mendeteksi outlier.
  • Heatmap digunakan untuk melihat korelasi antar fitur.

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Tahap data preparation dilakukan untuk memastikan data berada dalam kondisi optimal sebelum digunakan oleh model machine learning. Berikut ini adalah tahapan-tahapan data preparation yang dilakukan secara berurutan dan disesuaikan dengan kode implementasi:

**Rubrik/Kriteria Tambahan (Opsional)**:

- Exploratory Data Analysis (EDA) Sebelum memulai preprocessing, dilakukan eksplorasi awal untuk memahami kondisi data:
  • Menampilkan jumlah total data (df.shape)
  • Mengecek informasi struktur dan tipe data (df.info())
  • Menampilkan 5 sampel pertama data (df.head())
  • Mengecek missing value (df.isnull().sum())
  • Menampilkan statistik deskriptif (df.describe())
  • Visualisasi distribusi kelas (sns.countplot)
  • Visualisasi boxplot untuk mendeteksi outlier
  • Visualisasi histogram untuk melihat distribusi fitur
  • Heatmap korelasi antar fitur
- Alasan: EDA dilakukan untuk memahami karakteristik data, mengidentifikasi potensi masalah seperti outlier, distribusi tidak normal, korelasi antar variabel, serta membantu menentukan strategi preprocessing selanjutnya.

- Feature Engineering Dibuat beberapa fitur baru dari variabel yang sudah ada untuk memperkaya informasi yang diberikan ke model:
  • Glucose_Insulin_Ratio → membantu melihat rasio kadar gula terhadap insulin.
  • Glucose_BMI → menyoroti potensi hubungan obesitas dengan kadar gula darah.
  • Pregnancies_Age → mengindikasikan hubungan antara usia dan riwayat kehamilan terhadap risiko diabetes.
  • BMI_Category → pengelompokan berdasarkan standar WHO: Underweight, Normal, Overweight, Obese.
  • Age_Group → pengelompokan usia ke dalam Young, Middle, Old.
  Kemudian dilakukan one-hot encoding untuk fitur kategorik (BMI_Category, Age_Group) agar dapat digunakan oleh model machine learning.
- Alasan: Feature engineering membantu meningkatkan kapasitas model dalam menangkap pola yang tidak linear, interaksi antar fitur, dan memperluas wawasan model terhadap relasi risiko diabetes.

- Feature & Target Selection
  • X: semua fitur kecuali kolom target Outcome
  • y: target atau label klasifikasi (Outcome = 0 atau 1)
- Alasan: Memisahkan fitur dan label diperlukan agar pipeline machine learning dapat dijalankan dengan benar.

- Feature Scaling (Standardization)
  Dilakukan standardisasi terhadap seluruh fitur numerik menggunakan StandardScaler.

```
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- Alasan: Model seperti Logistic Regression, SVM, dan algoritma berbasis jarak sangat sensitif terhadap skala data. Scaling membuat semua fitur memiliki distribusi yang seragam, memudahkan model dalam belajar.

- Mengatasi Imbalanced Data dengan SMOTE
  Dataset memiliki distribusi kelas tidak seimbang. Untuk itu, dilakukan oversampling pada data latih menggunakan SMOTE (Synthetic Minority Over-sampling Technique).

```
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

```

- Alasan: Agar model tidak bias terhadap kelas mayoritas dan dapat mengenali kasus diabetes (kelas 1) dengan lebih baik.

- Hyperparameter Tuning
  Model Random Forest dituning menggunakan RandomizedSearchCV dengan beberapa parameter seperti:
  • n_estimators
  • max_depth
  • min_samples_split
  • min_samples_leaf
  • max_features
- Alasan: Tuning digunakan untuk mencari kombinasi parameter terbaik guna meningkatkan performa model terhadap data uji.

- Threshold Tuning (Optimasi Berdasarkan F1-score)
  Daripada menggunakan threshold default (0.5), dilakukan pencarian threshold optimal menggunakan precision-recall curve dan f1-score tertinggi.

```
precision, recall, thresholds = precision_recall_curve(y_test, proba)
```

- Alasan:
  Threshold default tidak selalu memberikan hasil terbaik, terutama untuk dataset tidak seimbang. Dengan threshold yang disesuaikan, model dapat fokus pada peningkatan recall atau f1-score sesuai konteks.

## Modeling

Pada tahap ini, dilakukan proses pemodelan untuk menyelesaikan permasalahan klasifikasi diabetes menggunakan pendekatan machine learning. Tujuan dari pemodelan ini adalah untuk memprediksi apakah seorang pasien menderita diabetes (positif) atau tidak (negatif) berdasarkan sejumlah fitur medis.

**Rubrik/Kriteria Tambahan (Opsional)**:

Kelebihan dan Kekurangan Setiap Algoritma

1. Logistic Regression

Kelebihan:
• Cepat dan efisien dalam proses pelatihan model.
• Mudah diinterpretasikan karena setiap koefisien dapat menunjukkan pengaruh fitur terhadap kelas target.
• Cocok digunakan sebagai baseline model untuk klasifikasi biner.

Kekurangan:
• Kurang mampu menangkap hubungan non-linear antar fitur.
• Performa menurun pada dataset yang kompleks dengan interaksi antar variabel yang tidak linier.

⸻

2. Random Forest

Kelebihan:
• Memberikan akurasi yang tinggi dan tahan terhadap overfitting.
• Mampu menangkap pola non-linear dan interaksi antar fitur dengan baik.
• Bekerja dengan baik pada dataset yang tidak seimbang (imbalanced data).
• Tidak memerlukan scaling data secara ketat.

Kekurangan:
• Cenderung lebih sulit untuk diinterpretasikan dibandingkan model linier.
• Membutuhkan proses tuning hyperparameter untuk mencapai performa terbaik.
• Model bisa cukup besar dan lambat untuk prediksi real-time jika jumlah pohon terlalu banyak.

⸻

3. XGBoost

Kelebihan:
• Sangat powerful dan sering menghasilkan performa terbaik dalam kompetisi machine learning.
• Mampu menangani missing values secara langsung tanpa perlu imputasi.
• Mendukung regularisasi untuk menghindari overfitting.
• Efisien dalam pemrosesan data skala besar.

Kekurangan:
• Waktu pelatihan lebih lama dibandingkan Random Forest dan Logistic Regression.
• Proses tuning lebih kompleks karena banyaknya parameter yang tersedia.
• Bisa overfitting jika tidak dikontrol dengan baik melalui validasi dan parameter regularisasi.

**Jelaskan proses improvement yang dilakukan**.
Pada model Random Forest, dilakukan proses RandomizedSearchCV untuk meningkatkan performa baseline. Beberapa parameter yang dituning meliputi:
• n_estimators: jumlah pohon pada hutan
• max_depth: kedalaman maksimal pohon
• min_samples_split: minimum jumlah data untuk membagi node
• min_samples_leaf: minimum data yang diperlukan di setiap daun
• max_features: jumlah fitur yang dipertimbangkan di tiap split

Tujuan tuning:
• Meningkatkan F1-score (metrik utama dalam project ini)
• Meningkatkan recall kelas 1 (positif diabetes) tanpa mengorbankan presisi secara drastis
• Mencari keseimbangan model agar tidak underfitting atau overfitting

Hasil tuning terbaik:

```
{'n_estimators': 300,
 'min_samples_split': 2,
 'min_samples_leaf': 2,
 'max_features': 'sqrt',
 'max_depth': 40}
```

**Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

- Pemilihan Model Terbaik Sebagai Solusi
  Setelah membandingkan Logistic Regression, Random Forest, dan XGBoost, model Random Forest dipilih sebagai model final terbaik berdasarkan alasan berikut:
  • Memiliki metrik evaluasi terbaik secara keseluruhan, terutama pada recall dan f1-score kelas 1.
  • Hasil tuning memberikan AUC Score = 0.822, F1-score kelas 1 = 0.70, dan Recall kelas 1 = 0.71.
  • Menggunakan feature engineering, SMOTE, scaling, hyperparameter tuning, dan threshold optimization yang secara keseluruhan membentuk pipeline yang solid.
  • Lebih stabil dan lebih mudah dituning dibandingkan XGBoost, dan memiliki performa lebih baik dari Logistic Regression pada dataset ini.

## Evaluation

Metrik Evaluasi yang Digunakan adalah Accuracy, Precision, Recall, F1-Score, AUC (Area Under Curve – ROC).

**Rubrik/Kriteria Tambahan (Opsional)**:

Karena proyek ini merupakan kasus klasifikasi biner (memprediksi apakah pasien menderita diabetes atau tidak), maka digunakan beberapa metrik evaluasi penting, yaitu: 
1. Accuracy
Mengukur proporsi prediksi yang benar terhadap seluruh data.
• Formula:
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
• Cocok digunakan saat distribusi kelas seimbang, tetapi kurang informatif jika dataset imbalanced, seperti pada kasus ini.
2. Precision
Mengukur seberapa akurat prediksi positif model.
• Formula:
\text{Precision} = \frac{TP}{TP + FP}
• Artinya: Dari semua yang diprediksi positif diabetes, berapa yang benar-benar positif. Berguna saat kita ingin meminimalkan false positive.
3. Recall (Sensitivity)
Mengukur kemampuan model dalam menemukan semua kasus positif.
• Formula:
\text{Recall} = \frac{TP}{TP + FN}
• Artinya: Dari semua pasien yang benar-benar positif diabetes, berapa yang berhasil terdeteksi oleh model.
• Dalam kasus ini, recall sangat penting, karena salah tidak mendeteksi pasien diabetes bisa berdampak serius.
4. F1-Score
Rata-rata harmonis antara precision dan recall.
• Formula:
\text{F1} = 2 \times \frac{Precision \times Recall}{Precision + Recall}
• Berguna untuk menyeimbangkan precision dan recall terutama saat distribusi data tidak seimbang.
5. AUC (Area Under Curve – ROC)
Mengukur kemampuan model dalam membedakan kelas 0 dan 1 secara keseluruhan.
• Nilai AUC berada pada rentang 0 hingga 1, dan semakin tinggi, semakin baik model dalam mengklasifikasikan.

⸻

📊 Hasil Evaluasi Model Terbaik (Random Forest + Feature Engineering + SMOTE + Hyperparameter + Threshold tuning)

Setelah melakukan serangkaian eksperimen dan tuning, berikut hasil akhir dari model Random Forest terbaik yang dipilih:
• Accuracy: 0.78
• Precision (kelas 1 - diabetes): 0.68
• Recall (kelas 1 - diabetes): 0.71
• F1-Score (kelas 1): 0.70
• AUC Score: 0.822

🔍 Interpretasi:

    •	Model berhasil menemukan 71% dari seluruh kasus diabetes (recall), yang merupakan metrik paling penting dalam konteks medis ini.
    •	Nilai AUC 0.822 menandakan bahwa model cukup baik dalam membedakan antara pasien dengan dan tanpa diabetes.
    •	Dengan threshold tuning otomatis berdasarkan F1-score, model berhasil mencapai keseimbangan antara ketepatan prediksi dan kemampuan deteksi.

🎯 Kesimpulan Evaluasi

Metrik evaluasi yang digunakan sudah sesuai dengan konteks permasalahan. Fokus utama adalah meningkatkan recall untuk mendeteksi sebanyak mungkin pasien yang berisiko diabetes, dengan tetap menjaga precision dan f1-score agar sistem tetap andal dan tidak terlalu banyak salah alarm (false positives).

Model yang dikembangkan tidak hanya menunjukkan performa tinggi di atas kertas, namun juga praktis dan dapat diimplementasikan dalam sistem deteksi dini risiko diabetes, khususnya untuk mendukung keputusan di sektor kesehatan masyarakat atau klinik.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
