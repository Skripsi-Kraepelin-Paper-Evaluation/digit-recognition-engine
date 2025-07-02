# PAPER BASED EVALUATOR KRAEPELIN
repository kode sistem untuk skripsi dari Hadekha dan Christian terkait sistem evaluasi Kraepelin berbasis kertas berbasis computer vision dan digit
recognition

## INFERENCE ENGINE
menggunakan pretrained model CNN yang dievaluasi menggunakan dataset MNIST dan EMNIST serta
dengan tambahan data augmentation. Setiap sampel training set telah diaugmentasi (zoom, shift, rotate) untuk menambah
akurasi dari model.
credit to : https://www.kaggle.com/models/pauljohannesaru/beyond_mnist

## STRUKTUR KODE

### dataservices
merepresentasikan kode terkait abstraksi pengelolaan dan pengambilan data

### controllers
merepresentasikan kode logic / function business process dari system

### models
merepresentasikan abstraksi objek / entitas untuk mempermudah abstraksi system

### engines
merepresentasikan mesin utama dari system seperti inference engine, preprocessing, dan region of interest

### persistent
direktori aset business process disimpan, seperti upload documents, rekam jejak evaluasi, dan gambar hasil teknik region of interest

## DISCLAIMER

this code is built with monolithic architecture in mind