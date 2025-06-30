import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import Tuple, List
import glob

class DatasetLoader:
    """Class untuk memuat dan memproses berbagai format dataset"""
    
    def __init__(self):
        self.datasets = {}
    
    def load_kana_mnist(self, x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memuat dataset Kana-MNIST dari file .npy
        
        Args:
            x_path: Path ke file X.npy
            y_path: Path ke file Y.npy
            
        Returns:
            Tuple dari (images, labels)
        """
        try:
            X = np.load(x_path)
            y = np.load(y_path)
            
            # Reshape jika diperlukan (pastikan format (N, 28, 28))
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], 28, 28, 1)
            elif len(X.shape) == 2:
                X = X.reshape(X.shape[0], 28, 28, 1)
            
            print(f"Kana-MNIST loaded: {X.shape[0]} samples")
            return X, y
            
        except Exception as e:
            print(f"Error loading Kana-MNIST: {e}")
            return None, None
    
    def load_emnist_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memuat dataset EMNIST dari file CSV
        
        Args:
            csv_path: Path ke file CSV EMNIST
            
        Returns:
            Tuple dari (images, labels)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Kolom pertama adalah label, sisanya pixel values
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
            
            # Reshape ke format gambar 28x28x1
            X = X.reshape(-1, 28, 28, 1)
            
            print(f"EMNIST loaded: {X.shape[0]} samples")
            return X, y
            
        except Exception as e:
            print(f"Error loading EMNIST: {e}")
            return None, None
    
    def load_nist_directory(self, root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memuat dataset NIST dari direktori dengan struktur folder berdasarkan kelas
        Struktur: root_dir/0/hsf_0/*.png, root_dir/0/hsf_1/*.png, ..., root_dir/9/hsf_7/*.png
        
        Args:
            root_dir: Path ke direktori root yang berisi folder 0-9
            
        Returns:
            Tuple dari (images, labels)
        """
        try:
            images = []
            labels = []
            
            # Iterate melalui setiap direktori kelas (0-9)
            for class_dir in sorted(os.listdir(root_dir)):
                class_path = os.path.join(root_dir, class_dir)
                
                if os.path.isdir(class_path) and class_dir.isdigit():
                    class_label = int(class_dir)
                    
                    # Iterate melalui subdirektori hsf_0 hingga hsf_7
                    for hsf_dir in sorted(os.listdir(class_path)):
                        hsf_path = os.path.join(class_path, hsf_dir)
                        
                        # Pastikan ini adalah direktori hsf_X
                        if os.path.isdir(hsf_path) and hsf_dir.startswith('hsf_'):
                            # Muat semua gambar PNG dari direktori hsf_X
                            image_files = glob.glob(os.path.join(hsf_path, "*.png"))
                            
                            for img_path in image_files:
                                try:
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is not None:
                                        # Resize ke 28x28 jika diperlukan
                                        if img.shape != (28, 28):
                                            img = cv2.resize(img, (28, 28))
                                        
                                        images.append(img)
                                        labels.append(class_label)
                                except Exception as e:
                                    print(f"Warning: Could not load image {img_path}: {e}")
                                    continue
                    
                    print(f"Loaded class {class_label}: {sum(1 for label in labels if label == class_label)} images")
            
            if not images:
                print("Warning: No images found in the specified directory structure")
                return None, None
            
            X = np.array(images).reshape(-1, 28, 28, 1)
            y = np.array(labels)
            
            print(f"NIST loaded: {X.shape[0]} total samples from {len(set(y))} classes")
            return X, y
            
        except Exception as e:
            print(f"Error loading NIST: {e}")
            return None, None
    
    def combine_datasets(self, datasets: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menggabungkan multiple datasets
        
        Args:
            datasets: List dari tuple (X, y)
            
        Returns:
            Combined (X, y)
        """
        valid_datasets = [(X, y) for X, y in datasets if X is not None and y is not None]
        
        if not valid_datasets:
            raise ValueError("No valid datasets to combine")
        
        X_combined = np.concatenate([X for X, y in valid_datasets], axis=0)
        y_combined = np.concatenate([y for X, y in valid_datasets], axis=0)
        
        print(f"Combined dataset: {X_combined.shape[0]} total samples")
        return X_combined, y_combined

class EfficientNetDigitClassifier:
    """EfficientNet model untuk klasifikasi digit"""
    
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def create_model(self, efficientnet_version='B0'):
        """
        Membuat model EfficientNet untuk klasifikasi digit
        
        Args:
            efficientnet_version: Versi EfficientNet ('B0', 'B1', dll.)
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape)
        
        # Konversi grayscale ke RGB (EfficientNet membutuhkan 3 channel)
        if self.input_shape[-1] == 1:
            x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        else:
            x = inputs
        
        # Resize input ke ukuran minimum yang diperlukan EfficientNet
        x = layers.UpSampling2D(size=(8, 8))(x)  # 28x28 -> 224x224
        
        # Load EfficientNet sebagai base model
        if efficientnet_version == 'B0':
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        elif efficientnet_version == 'B1':
            base_model = keras.applications.EfficientNetB1(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        else:
            raise ValueError(f"Unsupported EfficientNet version: {efficientnet_version}")
        
        # Freeze base model layers untuk transfer learning
        base_model.trainable = False
        
        # Add custom head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create final model
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"EfficientNet{efficientnet_version} model created successfully!")
        return self.model
    
    def fine_tune_model(self, learning_rate=0.0001):
        """
        Fine-tune model dengan membuka beberapa layer terakhir
        """
        if self.model is None:
            raise ValueError("Model must be created first")
        
        # Unfreeze beberapa layer terakhir dari base model
        base_model = self.model.layers[3]  # EfficientNet base model
        base_model.trainable = True
        
        # Freeze semua layer kecuali beberapa yang terakhir
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile dengan learning rate yang lebih kecil
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model ready for fine-tuning!")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Training model
        """
        if self.model is None:
            raise ValueError("Model must be created first")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_efficientnet_digit_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Training
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluasi model pada test set"""
        if self.model is None:
            raise ValueError("Model must be created first")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {loss:.4f}")
        
        return loss, accuracy
    
    def predict(self, X):
        """Prediksi untuk input data"""
        if self.model is None:
            raise ValueError("Model must be created first")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def preprocess_data(X, y):
    """
    Preprocessing data untuk training
    """
    # Normalisasi pixel values ke range [0, 1]
    X = X.astype('float32') / 255.0
    
    # Pastikan format yang benar
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    return X, y

def main():
    """
    Fungsi utama untuk menjalankan training
    """
    # Initialize dataset loader
    loader = DatasetLoader()
    
    # Load datasets (sesuaikan path dengan lokasi file Anda)
    print("Loading datasets...")
    
    # Kana-MNIST
    kana_X, kana_y = loader.load_kana_mnist('/home/hadekha/Downloads/kanada-mnist/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Dig_MNIST/X_dig_MNIST/arr_0.npy', '/home/hadekha/Downloads/kanada-mnist/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Dig_MNIST/y_dig_MNIST/arr_0.npy')
    
    # EMNIST
    emnist_X, emnist_y = loader.load_emnist_csv('/home/hadekha/Downloads/emnist/emnist-digits-train.csv')
    
    # NIST
    nist_X, nist_y = loader.load_nist_directory('/home/hadekha/Downloads/nist/by_class/')
    
    # Combine datasets
    datasets = [(kana_X, kana_y), (emnist_X, emnist_y), (nist_X, nist_y)]
    X_combined, y_combined = loader.combine_datasets(datasets)
    
    # Preprocess data
    X_combined, y_combined = preprocess_data(X_combined, y_combined)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train model
    classifier = EfficientNetDigitClassifier()
    
    # Create model
    model = classifier.create_model(efficientnet_version='B0')
    model.summary()
    
    # Initial training
    print("\nStarting initial training...")
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    
    # Fine-tuning
    print("\nStarting fine-tuning...")
    classifier.fine_tune_model(learning_rate=0.0001)
    history_fine = classifier.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=16)
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = classifier.evaluate(X_test, y_test)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save final model
    classifier.model.save('final_efficientnet_digit_model.h5')
    print("Model saved as 'final_efficientnet_digit_model.h5'")

if __name__ == "__main__":
    main()