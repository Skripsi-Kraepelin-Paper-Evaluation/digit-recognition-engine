from flask import Blueprint, jsonify, request
import os
import re
import concurrent.futures
from models import predicted_digit_answer as answer
from models import predicted_digit_question as question
import numpy as np
import psutil
import json
from configs import config

# Singleton for inference engine
class InferenceEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_engine(self, engine):
        self._engine = engine

    def predict_digit(self, is_answer,image_path, kernel_size=3, iterations=1):
        return self._engine.predict_digit(is_answer=is_answer,image_path=image_path, kernel_size=kernel_size, iterations=iterations)

class PredictHandler:
    def __init__(self, persistent_path='./persistent'):
        self.engine = InferenceEngine()
        self.persistent_path = persistent_path

    def _create_obj(self, predicted_digit, accuracy, row, col, blank, is_question):
        model_class = question.PredictedDigitQuestion if is_question else answer.PredictedDigitAnswer
        return model_class(
            digit=predicted_digit if not blank else None,
            accuracy=accuracy if not blank else None,
            column=col,
            row=row,
            need_manual_check=bool(accuracy < 0.8) if not blank else False,
            checked=False,
            is_blank=blank
        )

    def _serialize_obj(self, obj):
        return {
            'digit': int(obj.digit) if obj.digit is not None else None,
            'accuracy': float(obj.accuracy) if obj.accuracy is not None else None,
            'column': int(obj.column),
            'row': int(obj.row),
            'need_manual_check': bool(obj.need_manual_check),
            'checked': bool(obj.checked),
            'is_blank': bool(obj.is_blank)
        }

    def process_images(self, path, is_question=True):
        if not os.path.exists(path):
            return []

        images = [f for f in os.listdir(path) if f.endswith('.png')]

        def process_single_image(image):
            match = re.match(r'row(\d+)col(\d+)\.png', image)
            if not match:
                return None

            row, col = map(int, match.groups())
            image_path = os.path.join(path, image)

            try:
                predicted_digit, accuracy, blank = self.engine.predict_digit(
                    is_answer=not is_question,
                    image_path=image_path,
                    kernel_size=3,
                    iterations=1
                )
                return self._create_obj(predicted_digit, accuracy, row, col, blank, is_question)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                return None

        results = []
        max_workers = calculate_optimal_workers()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_image, image) for image in images]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        return results

    def handle_predict(self, filename):
        base_path = f'{self.persistent_path}/roi_result/{filename}'
        questions = self.process_images(os.path.join(base_path, 'questions'), is_question=True)
        answers = self.process_images(os.path.join(base_path, 'answers'), is_question=False)


        result = {
            'filename': filename,
            'questions': [self._serialize_obj(q) for q in questions],
            'answers': [self._serialize_obj(a) for a in answers],
            'total_questions': len(questions),
            'total_answers': len(answers)
        }

        ## save result as json to ./persistent/preview_history/{filename.json} so that it can be opened again

        preview_dir = f'{self.persistent_path}/preview_history'
        os.makedirs(preview_dir, exist_ok=True)
        
        # Save result as JSON file
        json_filename = f'{filename}.json'
        json_path = os.path.join(preview_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Add the saved path to the result for reference
        result['saved_path'] = json_path

        #####################################

        return result

def create_predict_blueprint(inference_engine,cfg):
    InferenceEngine().set_engine(inference_engine)
    predict_handler = PredictHandler(persistent_path=cfg.persistent_path)
    predict_bp = Blueprint('predict_controller', __name__)

    @predict_bp.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            filename = data.get('filename')
            if not filename:
                return jsonify({"error": "Missing filename"}), 400

            result = predict_handler.handle_predict(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return predict_bp

def calculate_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative estimates for CNN memory usage
    # Adjust these based on your specific model size
    estimated_model_memory_gb = 0.5  # Typical small CNN model
    max_memory_workers = int(memory_gb * 0.7 / estimated_model_memory_gb)  # Use 70% of RAM
    
    # CPU-based calculation
    cpu_workers = max(1, cpu_count - 1)  # Leave one core free
    
    # Take the minimum to avoid resource exhaustion
    optimal_workers = min(cpu_workers, max_memory_workers, 8)  # Cap at 8 for stability
    
    print(f"System specs:")
    print(f"  CPU cores: {cpu_count}")
    print(f"  RAM: {memory_gb:.1f} GB")
    print(f"  Calculated optimal workers: {optimal_workers}")
    
    return optimal_workers