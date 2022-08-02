## _**Graduation Rate**_


### Domain Proyek
Seringkali kita berpendapat bahwa kesempatan siswa sekolah untuk berprestasi sangat dipengaruhi oleh latar belakang keluarganya. Latar belakang yang dimaksud adalah tingkat pendidikan orang tua, tingkat penghasilan orang tua, asal suku, dan sebagainya. Hal ini telah ditunjukkan oleh beberapa studi yang menunjukkan bahwa latar belakang keluarga memang berpengaruh terhadap hasil belajar siswa di sekolah. Salah satunya adalah studi yang dilakukan oleh AJ Egalite dalam artikelnya yang berjudul ["How Family Background Influences Student Achievement"](https://www.uccronline.it/wp-content/uploads/2012/06/201604educationnext.pdf).
Berangkat dari hal tersebut, proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi masa studi mahasiswa berdasarkan data latar belakang keluarganya dan data kemampuan awalnya sebelum masuk kuliah. Mengapa model ini perlu dibuat? Agar pihak sekolah atau orang tua dapat memperoleh gambaran mengenai nilai akhir siswa dan dapat memberikan tindakan preventif jika nilai jelek yang muncul. Tindakan preventif dapat berupa tambahan jam pelajaran dari sekolah atau pemberian tutor sebaya untuk siswa yang berisiko memperoleh nilai buruk. 

### Business Understanding 
Dari uraian pada bagian sebelumnya, dapat dirumuskan permasalahan yang akan diselesaikan yaitu 
- Bagaimana model machine learning yang dapat melakukan prediksi masa studi mahasiswa berdasarkan data latar belakang orang tua dan kemampuan awal mahasiswa?

Dengan demikian, tujuan dari proyek ini adalah 
- Untuk mengembangkan model machine learning yang dapat melakukan prediksi masa studi mahasiswa berdasarkan latar belakang orang tua dan kemampuan awal mahasiswa

#### Solution Statement
Untuk dapat mencapai tujuan di atas, kita dapat: 
- Menggunakan algoritma klasifikasi untuk mengembangkan model, kemudian melakukan hyperparameter tuning untuk mencari model terbaik. Penentuan model terbaik menggunakan F1 Score. Jika F1 Score mencapai lebih dari 0.75 maka model dianggap cukup baik.

### Data Understanding
Data yang digunakan pada proyek ini adalah data graduation rate dari blog [Royce Kimmons](http://roycekimmons.com/tools/generated_data/graduation_rate). Kumpulan data ini mencakup tingkat kelulusan dari perguruan tinggi empat tahun dengan berbagai faktor siswa.

![info](https://i.ibb.co/jGQL1ZF/Screenshot-54.png)

Dari data diatas dapat diketahui terdapat 7 buah atribut, 1000 baris data dan tidak terdapat missing values. 
adapun deskripsi data tersebut adalah :
- Atribut ACT composite score, yaitu merupakan rata-rata dari 4 tes utama yaitu English, Reading, Math, and Science, yaitu merupakan nilai ujian yang mengukur kemampuan berpikir matematis dan menalar. 
- Parental level of education, yaitu tingkat pendidikan orang tua. 
- Parental income, yaitu penghasilan orang tua dalam US dollar.
- High school gpa, yaitu nilai akhir ketika SMA. 
- College gpa, yaitu IPK terakhir.
- Years to graduate, yaitu masa studi dalam satuan tahun. 

__Exploratory Data Analysis__
Berikut adalah visualisasi data untuk masing-masing atribut yang digunakan. 
![Visualisasi](https://github.com/nurbenasution/dicoding/blob/fab7bb30b986ff6516dfb938622d829b27e073a8/visualisasi.png?raw=true)

_Sebaran Statistik Setiap Atribut_
![Stat](https://i.ibb.co/hmCpsZt/Screenshot-56.png)

_Grafik Sebaran Parental Level of Education_ 

![edu](https://i.ibb.co/jTzFTjp/Screenshot-58.png) 

Dari visualisasi & Sebaran Statistik diatas dapat disimpulkan :
-  ACT composite score
  Nilai minimal 21 dan maksimum 36 serta mean 28.71
- Parental income
  Nilai minimum $11248 dan nilai maksimum $116611 serta mean 66471
- High school gpa
  Nilai minimal 2.6 dan nilai maksimal 4 serta mean 3.73
- College gpa
  Nilai minimal 2.6 dan maksimal 4 serta mean 3.38
- Years to graduate
  Nilai minimal 3 dan maksimal 10 serta mean 4.95
- Parental Level of Education
  'some college' merupakan level education terbanyak dari orang tua mahasiswa, kemudian yang paling sedikit adalah 'Masters degree'

### Data Preparation
Pada tahapan ini, data disiapkan untuk selanjutnya dapat diolah untuk menyelesaikan permasalahan. 
##### Tranformasi Data

Untuk dapat memprediksi masa studi mahasiswa, terdapat beberapa hal yang harus dilakukan, yaitu

- Pihak universitas hanya penasaran apakah seorang mahasiswa akan lulus tepat waktu atau tidak. Dengan demikian, variabel years to graduate perlu diubah menjadi variabel yang bersifat kategorikal dengan 2 nilai yaitu lulus tepat waktu (on time) dan tidak tepat waktu (late). Dengan asumsi bahwa siswa S1 seharusnya menempuh studi selama 5 tahun, maka dikatakan "On Time" jika nilai years to graduate <=5 dan dikatakan "Late" jika years to graduate > 5. 
- Setelah diubah ke bentuk kategorik, agar dapat dimodelkan maka variabel ini kemudian diubah ke data numerik yaitu dengan aturan 1 --> on Time, 2 --> Late.
- Variabel parental education merupakan variabel kategorik sehingga perlu diubah ke dalam bentuk numerik yaitu 'some high school':0, 'high school':1,'some college':2,'bachelor\'s degree':3,'associate\'s degree':4,'master\'s degree':5

##### Train-Test-Split
Membagi dataset menjadi data latih (train) dan data uji (test) dengan menggunakan proporsi pembagian sebesar 70:30 dengan fungsi train_test_split dari sklearn.
### Modeling

Selanjutnya adalah tahap modeling. Perlu diketahui karena pada proyek ini variabel terikat yang terlibat adalah kategori masa studi mahasiswa (years category) yang hanya memiliki 2 nilai, maka proyek ini termasuk masalah klasifikasi. Adapun algoritma yang kita digunakan untuk masalah klasifikasi adalah K-Nearest Neighbours.

berikut nilai F1-score K Nearest Neighbours: 0.8042105263157895

kemudian dioptimalkan dengan Hyperparameter, yang didapatkan parameter
knn__n_neighbors: 8 dan knn__weights : uniform

lalu dimasukan ke algoritma knn_tuned dan didapatkan 
F1-score K Nearest Neighbours Tuned: 0.831013916500994
sehingga terjadi kenaikan 0.03 dari nilai K-Nearest Neighbours awal.

### Evaluation
Metrik yang dipilih adalah F1 Score, karena meskipun ini bukan masalah hidup dan mati, tetapi mengingat bagi mahasiswa yang sudah diprediksi tidak akan lulus tepat waktu akan diberikan perlakuan dari pihak kampus, maka ketepatan model dalam memprediksi sangatlah penting. Bayangkan jika mahasiswa yang sebenarnya akan lulus tepat waktu tetapi diprediksi lulus terlambat (false negatif) terlanjur diberikan perlakuan dari kampus maka hal ini merugikan sumber daya kampus. 
Lebih lanjut, bayangkan jika mahasiswa yang sebenarnya akan lulus terlambat tetapi diprediksi lulus tepat waktu (false positif) maka tindakan preventif dari kampus tidak akan berguna. Dengan demikian, dalam masalah ini, sebisa mungkin kita memilih model dimana false positif dan false negatifnya rendah, sehingga metrik evaluasi yang digunakan adalah F1 Score. 

Dari modeling dengan K Nearest Neighbours yang telah dibuat menghasilkan F1-score  yang telah ditune sebesar 0.831013916500994
dan dibawah ini adalah tampilan confusion matrix untuk model KNN yang telah di tune  

![Confusion Matrix KNN Tuned](https://i.ibb.co/JF7F68K/Screenshot-57.png)

Sedangkan berikut adalah cuplikan gambar dari [web ini](https://www.teknologi-bigdata.com/2020/05/validitas-rapid-test-covid-19-akurasi-accuracy-vs-f1-score.html#:~:text=F1%2Dscore%20digunakan%20ketika%20False,True%20Positive%20dan%20True%20Negative), yang menunjukkan cuplikan formula dari F1 Score. 
![Formula F1 Score](https://github.com/nurbenasution/dicoding/blob/985ffdd859e8e4700c33e2d21ff057ce8345e2bc/formula%20F1%20Score.png?raw=true)

 dimana P merupakan Precision dan R merupakan Recall. 
 Precision menunjukkan perbandingan antara jumlah yang benar positif (TP) dengan jumlah seluruh diagnosa/prediksi positif (TP dan FP). Sehingga jika Precision bernilai besar maka True Positif Besar dan False Positif Kecil. 
 Sedangkan, Recall menunjukkan perbandingan antara jumlah yang benar positif (TP) dengan jumlah yang pada kenyataannya positif. Sehingga jika Recall bernilai besar maka True Positif Besar dan False Negatif Kecil. 
 
 Dengan demikian, jika F1 Score besar maka Precision atau Recall bernilai besar atau dengan kata lain nilai False Positif atau False Negatif kecil. 



