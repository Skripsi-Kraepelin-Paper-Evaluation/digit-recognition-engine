import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class MNISTInference:
    def __init__(self, model_path='./output_model/mnist_digit_classifier.h5'):
        """
        Initialize the inference class with a trained model
        
        Args:
            model_path (str): Path to the saved model directory
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess PNG image to match MNIST training format
        
        Args:
            image_path (str): Path to the PNG image
            
        Returns:
            numpy.ndarray: Preprocessed image ready for prediction
        """
        # Load the image
        img = Image.open(image_path)
        
        # Convert to grayscale if it's not already
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28 pixels (MNIST size)
        img = img.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # MNIST digits are white on black background
        # If your image has black digits on white background, invert it
        # Check if background is mostly white (average > 127)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize pixel values to 0-1 range
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape to match model input: (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def predict_digit(self, image_path, show_image=True):
        """
        Predict digit from PNG image
        
        Args:
            image_path (str): Path to the PNG image
            show_image (bool): Whether to display the image
            
        Returns:
            tuple: (predicted_digit, confidence, all_probabilities)
        """
        # Preprocess the image
        processed_img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Get the predicted class and confidence
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit]
        
        # Display the image if requested
        if show_image:
            plt.figure(figsize=(8, 4))
            
            # Original image
            plt.subplot(1, 2, 1)
            original_img = Image.open(image_path)
            plt.imshow(original_img, cmap='gray')
            plt.title(f'Original Image')
            plt.axis('off')
            
            # Processed image
            plt.subplot(1, 2, 2)
            plt.imshow(processed_img.reshape(28, 28), cmap='gray')
            plt.title(f'Processed (28x28)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Show prediction probabilities
            plt.figure(figsize=(10, 6))
            digits = range(10)
            plt.bar(digits, predictions[0])
            plt.xlabel('Digit')
            plt.ylabel('Probability')
            plt.title(f'Prediction Probabilities - Predicted: {predicted_digit} (Confidence: {confidence:.3f})')
            plt.xticks(digits)
            plt.show()
        
        return predicted_digit, confidence, predictions[0]
    
    def predict_batch(self, image_paths):
        """
        Predict digits from multiple PNG images
        
        Args:
            image_paths (list): List of paths to PNG images
            
        Returns:
            list: List of tuples (image_path, predicted_digit, confidence)
        """
        results = []
        
        for image_path in image_paths:
            try:
                predicted_digit, confidence, _ = self.predict_digit(image_path, show_image=False)
                results.append((image_path, predicted_digit, confidence))
                print(f"{os.path.basename(image_path)}: Predicted = {predicted_digit}, Confidence = {confidence:.3f}")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append((image_path, None, None))
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the inference class
    inferencer = MNISTInference('./output_model/mnist_digit_classifier.keras')
    
    # Example 1: Single image prediction
    # Replace 'path_to_your_image.png' with actual path
    try:
        image_path = 'test_image/0.png'  # Change this to your image path
        digit, confidence, probabilities = inferencer.predict_digit(image_path)
        print(f"Predicted digit: {digit}")
        print(f"Confidence: {confidence:.3f}")
        print(f"All probabilities: {probabilities}")
    except FileNotFoundError:
        print("Image file not found. Please provide a valid path to a PNG image.")