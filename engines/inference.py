import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
import os
import cv2
from scipy import ndimage
from skimage import filters

class NewDigitsRecogModel:
    def __init__(self, model_path,min_threshold_zero=50):
        """
        Initialize the inference class with a trained model

        Args:
            model_path (str): Path to the saved model directory
        """
        self.model = tf.keras.models.load_model(model_path)
        self.min_threshold_zero = min_threshold_zero
        print(f"Model loaded from {model_path}")

    def gaussian_blur_thickening(self, image, sigma=1.2, threshold=80):
        """
        Use Gaussian blur followed by thresholding to thicken digit lines

        Args:
            image: Input image - white digits on black background
            sigma: Standard deviation for Gaussian blur (increased)
            threshold: Threshold for binarization after blur (lowered)
        """
        # Ensure proper format
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Apply Gaussian blur to spread the white pixels
        blurred = filters.gaussian(image.astype(np.float32), sigma=sigma)

        # Convert back to 0-255 range
        blurred = (blurred * 255 / blurred.max()).astype(np.uint8)

        # Lower threshold to capture more of the blurred digit
        thickened = (blurred > threshold).astype(np.uint8) * 255

        return thickened

    def preprocess_image(self, image_path, **enhancement_kwargs):
        """
        Preprocess PNG image to match MNIST training format with improved line thickness enhancement

        Args:
            image_path (str): Path to the PNG image
            **enhancement_kwargs: Additional parameters for enhancement method

        Returns:
            tuple: (preprocessed_image, processing_steps_dict)
        """

        # Blank image detection
        imgcv = cv2.imread(image_path)
        gray = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # Invert: digits become white
        non_zero = cv2.countNonZero(thresh)
        
        is_blank = False

        if non_zero < self.min_threshold_zero: 
            is_blank = True
            return None, is_blank

        # Load the image
        img = Image.open(image_path)

        # Convert to grayscale if it's not already
        if img.mode != 'L':
            img = img.convert('L')

        # Store original for visualization
        original_img = np.array(img)

        # Resize to larger size first to preserve details during enhancement
        img_large = img.resize((56, 56), Image.LANCZOS)  # 2x target size

        img_array = np.array(img_large)
        

        # MNIST digits are white on black background
        # If your image has black digits on white background, invert it
        # Check if background is mostly white (average > 127)
        inverted = False
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            inverted = True

        # Apply line thickness enhancement
        enhanced_img = img_array.copy()

        sigma = enhancement_kwargs.get('sigma', 1.2)  # Increased default
        threshold = enhancement_kwargs.get('threshold', 80)  # Lowered default
        enhanced_img = self.gaussian_blur_thickening(img_array, sigma, threshold)

        # Resize to final MNIST size (28x28)
        enhanced_pil = Image.fromarray(enhanced_img)
        final_img = enhanced_pil.resize((28, 28), Image.LANCZOS)
        final_array = np.array(final_img)

        # Normalize pixel values to 0-1 range
        final_array = final_array.astype('float32') / 255.0

        # Reshape to match model input: (1, 28, 28, 1)
        final_array = final_array.reshape(1, 28, 28, 1)

        return final_array, is_blank

    def predict_digit(self, image_path, **enhancement_kwargs):
        """
        Predict digit from PNG image with line thickness enhancement

        Args:
            image_path (str): Path to the PNG image
            show_image (bool): Whether to display the image and processing steps
            **enhancement_kwargs: Additional parameters for enhancement

        Returns:
            tuple: (predicted_digit, confidence, all_probabilities)
        """
        # Preprocess the image with enhancement
        processed_img, is_blank = self.preprocess_image(
            image_path,
            **enhancement_kwargs
        )

        if is_blank:
            return None, None, is_blank

        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)

        # Get the predicted class and confidence
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit]

        return predicted_digit, confidence, is_blank