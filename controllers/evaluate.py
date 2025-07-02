from flask import Blueprint, jsonify, request
import os
import re
from models import predicted_digit_answer as answer
from models import predicted_digit_question as question
import numpy as np

# Singleton for inference engine
class InferenceEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_engine(self, engine):
        self._engine = engine

    def predict_digit(self, image_path, kernel_size=3, iterations=1):
        return self._engine.predict_digit(image_path=image_path, kernel_size=kernel_size, iterations=iterations)

class EvalHandler:
    def __init__(self):
        self.engine = InferenceEngine()

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

        results = []
        images = [f for f in os.listdir(path) if f.endswith('.png')]

        for image in images:
            match = re.match(r'row(\d+)col(\d+)\.png', image)
            if not match:
                continue

            row, col = map(int, match.groups())
            image_path = os.path.join(path, image)

            try:
                predicted_digit, accuracy, blank = self.engine.predict_digit(
                    image_path=image_path,
                    kernel_size=3,
                    iterations=1
                )
                obj = self._create_obj(predicted_digit, accuracy, row, col, blank, is_question)
                results.append(obj)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        return results

    def handle_eval(self, filename):
        base_path = f'./persistent/roi_result/{filename}'
        questions = self.process_images(os.path.join(base_path, 'questions'), is_question=True)
        answers = self.process_images(os.path.join(base_path, 'answers'), is_question=False)

        return {
            'filename': filename,
            'questions': [self._serialize_obj(q) for q in questions],
            'answers': [self._serialize_obj(a) for a in answers],
            'total_questions': len(questions),
            'total_answers': len(answers)
        }

def create_eval_blueprint(inference_engine):
    InferenceEngine().set_engine(inference_engine)
    eval_handler = EvalHandler()
    eval_bp = Blueprint('eval_controller', __name__)

    @eval_bp.route('/eval', methods=['POST'])
    def eval():
        try:
            data = request.get_json()
            filename = data.get('filename')
            if not filename:
                return jsonify({"error": "Missing filename"}), 400

            result = eval_handler.handle_eval(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return eval_bp
