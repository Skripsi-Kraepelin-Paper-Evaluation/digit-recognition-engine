from flask import Blueprint, jsonify, request
import os
import re
import concurrent.futures
from models import predicted_digit_answer as answer
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
    def __init__(self, persistent_path='./persistent',min_accuracy=0.8):
        self.engine = InferenceEngine()
        self.persistent_path = persistent_path
        self.min_accuracy = min_accuracy

    def _create_obj(self, predicted_digit, accuracy, row, col, blank, is_question):
        model_class = answer.PredictedDigitAnswer
        return model_class(
            digit=predicted_digit if not blank else None,
            accuracy=accuracy if not blank else None,
            column=col,
            row=row,
            need_manual_check=bool(accuracy < self.min_accuracy) if not blank else False,
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
        # questions is fixed
        # 40 x 56
        questions_fixed = [
            [9,7,2,9,5,6,6,7,3,6,4,8,3,9,9,5,4,4,9,2,2,6,8,5,5,7,8,9,3,3,4,0,8,2,7,4,3,6,5,9,5,8,6,4,9,3,4,6,7,9,6,9,2,7,5,9],
            [9,9,8,4,4,6,5,2,7,7,8,9,3,3,7,1,6,2,5,7,9,4,2,8,2,7,7,3,5,2,3,2,4,7,5,8,6,9,8,4,5,3,7,4,6,3,8,8,7,6,5,9,8,9,4,4],
            [0,3,6,8,3,4,5,7,6,3,5,9,4,2,4,7,9,2,5,6,7,4,3,2,6,9,5,8,8,7,5,3,8,6,6,4,9,7,2,8,5,5,4,8,3,9,9,6,8,2,2,0,6,3,3,8],
            [4,2,6,4,3,7,9,6,5,4,8,6,7,7,2,3,3,4,9,5,2,7,6,8,8,9,2,4,5,3,9,8,5,5,6,3,2,9,4,4,6,2,2,8,7,5,9,9,7,3,8,4,6,2,3,4],
            [6,2,7,9,5,5,7,8,5,4,9,7,7,2,4,3,8,6,9,2,2,8,7,3,9,4,4,5,6,6,3,2,9,8,8,3,5,1,7,8,2,5,3,6,6,9,3,5,8,4,7,6,7,2,5,9],
            [8,8,4,5,3,8,7,7,2,9,6,5,4,4,7,6,2,8,9,9,3,2,0,2,5,2,3,6,4,8,2,6,7,4,6,5,9,6,8,9,9,3,4,7,5,8,4,2,3,3,7,8,4,8,3,5],
            [7,8,9,2,2,3,7,1,7,4,2,5,9,7,3,6,9,4,6,8,3,3,5,7,3,4,8,5,6,3,7,8,2,7,5,5,8,6,4,9,2,6,6,7,9,5,2,2,4,3,9,7,9,8,2,2],
            [7,2,6,8,8,3,4,8,3,5,2,7,6,4,3,6,5,3,9,8,4,7,5,4,2,9,6,6,3,8,5,5,9,7,4,6,9,3,3,2,8,7,9,9,5,6,2,4,4,5,7,7,6,2,8,8],
            [7,8,8,6,3,9,3,4,6,8,5,5,2,7,4,4,9,7,6,6,2,3,5,4,7,7,2,9,4,5,3,3,6,9,9,8,4,2,8,7,3,8,0,4,9,4,8,2,5,8,6,7,8,8,3,6],
            [2,8,6,4,2,7,7,8,3,9,5,4,4,6,7,3,3,5,2,9,8,8,4,3,6,9,1,9,5,9,6,7,5,6,3,2,5,7,9,6,4,8,9,2,6,5,8,2,2,4,3,2,6,8,2,4],
            [4,4,3,5,6,6,7,3,2,2,9,8,4,0,3,4,8,2,4,9,7,6,2,3,7,2,2,5,6,5,7,4,5,8,7,9,2,6,6,8,5,9,3,8,9,9,4,7,5,5,3,4,3,4,6,5],
            [2,4,8,9,7,5,7,9,5,2,6,8,3,7,6,5,9,2,7,8,6,9,9,3,6,2,3,4,5,8,8,2,5,4,6,3,3,8,7,7,4,9,6,4,7,2,8,5,5,3,9,2,8,4,7,9],
            [6,4,3,7,4,9,7,5,8,3,3,5,9,2,6,3,8,9,4,4,7,6,2,4,6,6,7,8,8,2,9,5,5,2,7,3,9,9,8,6,5,7,7,9,3,2,2,8,4,1,4,6,3,4,4,7],
            [6,6,8,2,6,4,9,7,7,5,3,6,2,8,8,3,9,5,5,9,3,3,7,6,5,4,4,8,7,2,2,9,8,5,0,5,4,2,3,4,5,8,7,2,5,3,6,8,5,6,9,6,8,6,6,2],
            [5,5,6,2,3,3,9,8,2,7,7,8,4,4,5,3,7,2,2,9,1,8,6,3,5,7,8,9,2,3,4,3,2,5,6,7,4,5,2,4,7,3,8,4,6,9,4,2,7,9,9,5,6,5,3,2],
            [2,9,8,0,8,7,5,8,3,4,8,9,9,7,3,5,4,7,9,4,2,8,6,3,2,4,6,9,2,5,7,6,8,8,9,3,6,6,5,9,6,7,4,3,8,8,5,2,6,4,9,2,8,9,8,0],
            [4,3,8,2,4,5,7,6,5,4,6,9,2,3,3,4,2,7,8,6,3,5,9,6,6,2,5,8,8,7,4,9,5,5,6,8,4,4,7,5,3,9,9,7,3,2,2,8,3,7,7,4,8,3,4,2],
            [5,5,6,7,2,8,9,2,6,4,8,2,3,6,6,8,4,3,5,4,9,8,8,7,3,2,2,9,4,6,2,7,7,8,5,3,9,9,5,1,5,2,6,4,8,5,9,3,6,7,9,5,6,5,2,7],
            [5,3,3,9,6,6,4,3,2,7,7,8,2,4,4,6,3,8,8,7,5,9,9,2,0,2,4,5,8,3,3,8,6,5,2,9,7,4,2,5,7,9,6,3,4,4,7,6,9,3,7,5,3,3,6,9],
            [7,6,6,9,5,1,6,2,8,6,3,8,2,5,8,3,6,8,4,7,6,5,4,2,5,9,7,3,4,8,9,5,7,9,3,5,5,6,9,4,5,2,6,7,2,2,3,7,4,9,8,7,6,6,5,9],
            [8,4,2,2,8,7,5,4,3,9,3,5,6,7,3,3,4,5,2,9,8,6,4,7,9,2,3,6,8,9,4,4,8,3,7,7,4,6,5,8,8,2,4,9,9,7,8,5,5,3,2,8,2,4,8,2],
            [4,3,6,6,8,5,2,6,9,5,7,6,2,9,7,3,5,5,9,6,7,2,4,6,3,8,9,9,4,2,2,3,4,8,7,7,5,3,9,8,8,2,0,9,6,2,6,7,2,5,7,4,6,3,8,6],
            [3,9,8,6,5,4,9,6,6,4,3,3,8,7,4,6,3,2,5,5,7,8,2,9,9,7,1,4,9,5,6,4,8,4,7,9,3,7,9,2,8,6,5,4,4,5,8,3,3,2,7,3,8,9,5,6],
            [2,4,7,6,6,5,8,3,3,2,9,6,0,7,2,6,9,2,5,9,8,3,4,8,5,3,6,8,8,9,3,7,5,2,4,4,5,8,4,2,7,3,5,6,9,4,7,7,6,2,2,2,7,4,6,6],
            [4,2,3,4,8,9,2,7,2,6,3,7,9,3,5,7,8,5,3,8,8,6,4,9,5,6,7,4,4,3,6,8,4,5,2,8,7,5,5,9,4,6,2,7,7,3,9,9,7,8,2,4,3,2,8,4],
            [9,7,9,9,3,6,7,3,8,2,2,3,5,4,6,5,2,6,9,4,7,6,6,8,5,7,4,9,5,5,8,6,3,3,6,2,8,4,4,3,7,7,5,3,2,9,8,8,7,1,5,9,9,7,3,9],
            [7,8,5,3,9,7,4,3,2,2,6,3,8,9,3,5,5,2,9,8,8,6,4,8,3,3,7,5,9,9,4,4,2,0,6,2,4,2,5,6,4,7,2,4,8,9,6,8,3,4,5,7,5,8,9,3],
            [9,4,5,3,7,8,2,2,7,5,5,8,6,3,3,2,9,7,7,4,1,8,4,3,6,6,4,7,6,8,7,2,5,7,9,6,5,4,9,5,8,4,6,9,2,7,3,4,5,6,7,9,5,4,7,3],
            [3,2,8,8,6,0,9,6,8,3,4,2,5,6,9,7,3,5,4,6,7,9,5,2,4,4,8,9,3,6,6,5,7,6,2,3,8,5,9,2,8,8,7,2,6,4,9,9,8,4,3,3,8,2,6,8],
            [6,5,4,9,2,2,3,3,4,5,3,6,6,2,5,8,3,7,9,8,5,5,6,4,8,2,7,3,5,2,4,3,8,7,6,3,9,4,4,7,8,9,5,7,7,4,6,7,2,9,9,6,4,5,2,9],
            [9,8,7,3,4,6,9,4,2,4,6,8,9,2,5,4,3,3,7,6,5,3,9,3,4,5,7,8,2,2,4,8,7,5,2,7,6,2,6,9,0,2,8,8,9,3,5,7,7,8,4,4,9,4,6,8],
            [5,4,2,7,4,3,5,5,9,9,5,6,7,4,8,6,3,8,5,2,8,3,6,8,2,6,1,5,9,6,6,7,2,3,5,5,8,7,9,9,4,2,8,8,5,6,4,7,7,3,8,5,8,9,3,4],
            [0,8,2,4,9,5,3,8,2,3,9,5,7,8,8,3,6,4,4,2,8,7,7,2,3,4,6,6,9,3,3,5,8,9,4,7,3,2,2,7,6,2,5,4,9,6,5,5,3,9,7,8,3,3,5,8],
            [3,4,7,2,8,9,6,7,2,2,8,9,4,5,3,4,8,6,6,3,2,8,4,6,2,9,8,2,7,6,5,5,7,3,9,6,7,4,4,3,6,9,7,5,2,4,7,9,2,5,6,7,6,3,9,4],
            [6,8,6,5,9,8,7,9,2,4,5,3,6,8,7,2,4,3,3,2,9,6,4,5,6,7,5,4,2,8,3,4,9,7,6,3,9,5,8,4,6,2,5,1,5,9,9,3,5,8,7,9,7,4,8,8],
            [2,2,3,6,6,9,5,6,4,7,8,2,4,9,7,4,5,3,7,9,9,8,4,3,8,5,7,8,0,8,9,2,7,7,3,8,2,2,3,7,9,9,3,5,7,4,4,8,6,5,5,6,5,7,9,2],
            [5,2,3,8,6,9,3,3,7,4,8,1,9,2,2,7,3,5,4,4,8,7,7,2,8,9,3,3,2,6,5,5,9,4,6,2,5,8,8,3,4,7,6,9,5,6,6,3,9,8,8,3,3,4,9,2],
            [5,7,6,4,7,8,2,9,4,2,3,3,9,5,5,9,3,8,8,2,6,3,5,7,7,9,4,6,2,8,6,6,9,9,7,2,4,9,6,4,8,3,7,4,2,5,4,7,6,5,2,9,2,2,8,7],
            [6,8,4,7,2,8,2,5,6,9,6,7,4,4,9,8,3,6,2,9,5,3,3,8,5,7,9,4,7,3,4,6,9,6,5,8,6,3,5,2,7,8,5,4,3,2,4,5,0,5,8,5,2,9,8,8],
            [6,5,3,3,9,4,8,2,2,8,9,6,8,7,2,9,5,6,7,3,8,6,2,5,9,7,5,7,9,8,4,2,4,1,4,8,2,2,3,9,7,7,5,6,8,9,9,3,7,2,5,2,8,8,4,5]            
        ]
        answers = self.process_images(os.path.join(base_path, 'answers'), is_question=False)
        answers_sorted = sorted(answers, key=lambda x: (x.column, x.row))

        # sort answers by first column, first rows


        result = {
            'filename': filename,
            'questions': questions_fixed,
            'answers': [self._serialize_obj(a) for a in answers_sorted],
            'total_questions': len(questions_fixed),
            'total_answers': len(answers_sorted)
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
    predict_handler = PredictHandler(persistent_path=cfg.persistent_path,min_accuracy=cfg.min_accuracy)
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