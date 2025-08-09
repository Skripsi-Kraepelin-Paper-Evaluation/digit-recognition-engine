import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from engines import inference  # Pastikan file/class ini benar

def load_images_from_directory(dataset_dir, max_per_class=100):
    image_paths = []
    true_labels = []

    for label_str in sorted(os.listdir(dataset_dir)):
        label_path = os.path.join(dataset_dir, label_str)
        if not os.path.isdir(label_path):
            continue

        all_images = [
            os.path.join(label_path, f)
            for f in os.listdir(label_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Acak dan ambil maksimal 100 gambar
        random.shuffle(all_images)
        selected_images = all_images[:max_per_class]

        image_paths.extend(selected_images)
        true_labels.extend([int(label_str)] * len(selected_images))

    return image_paths, true_labels

def evaluate_model(model_path, dataset_dir):
    # Inisialisasi model
    recognizer = inference.NewDigitsRecogModel(model_path)

    # Load gambar dengan batasan per kelas
    image_paths, true_labels = load_images_from_directory(dataset_dir, max_per_class=100)

    predictions = []
    final_labels = []

    for img_path, true_label in zip(image_paths, true_labels):
        predicted_digit, confidence, is_blank = recognizer.predict_digit(
            is_answer=True,
            image_path=img_path
        )

        if is_blank:
            print(f"[SKIP] Blank image detected: {img_path}")
            continue

        predictions.append(predicted_digit)
        final_labels.append(true_label)

    # Confusion matrix
    cm = confusion_matrix(final_labels, predictions)
    labels = list(range(10))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(final_labels, predictions, digits=4))

    # One-vs-Rest Evaluation
    print("\nOne-vs-Rest Metrics per Class:")
    mcm = multilabel_confusion_matrix(final_labels, predictions, labels=labels)

    for idx, matrix in enumerate(mcm):
        tn, fp, fn, tp = matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print(f"Class {idx}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  → Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

if __name__ == "__main__":
    model_path = "./output_model/model0.h5"  # Ubah sesuai lokasi model
    dataset_dir = "/home/hadekha/Pictures/SKRIPSI"  # Ubah sesuai dataset
    evaluate_model(model_path, dataset_dir)
