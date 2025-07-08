from flask import Blueprint, jsonify, request
import os
import json
from models import evaluator as ev

class SaveAndEvaluateHandler:
    def __init__(self,cfg):
        self.persistent_path = cfg.persistent_path
        self.cfg = cfg

    def handle_save_and_evaluate(self, filename):
        ## parsing data from payload
        try:
            payload_data = request.get_json()
            if not payload_data:
                return {
                    'error': 'No JSON payload provided',
                    'message': 'Request body must contain JSON data'
                }
            
            # Parse JSON data from payload
            if isinstance(payload_data, str):
                data = json.loads(payload_data)
            else:
                data = payload_data
                
        except Exception as e:
            return {
                'error': 'Invalid JSON payload',
                'message': str(e)
            }
        
        try:
            evaluator = ev.create_evaluator_from_json(data,self.cfg)
        except Exception as e:
            return {
                'error': 'Failed to create evaluator',
                'message': str(e)
            }

        ## generate correct results array
        try:
            results = evaluator.evaluate()
        except Exception as e:
            return {
                'error': 'Evaluation failed',
                'message': str(e)
            }

        ## evaluate kraepelin score
        overall_stats = evaluator.get_statistics(results)
        column_stats = evaluator.get_column_statistics(results)
        
        # Calculate Kraepelin specific metrics
        kraepelin_metrics = self.calculate_kraepelin_metrics(results, column_stats)
        
        # Prepare final result
        result = {
            'filename': filename,
            'evaluation_results': self.serialize_results(results),
            'overall_statistics': overall_stats,
            'column_statistics': column_stats,
            'kraepelin_metrics': kraepelin_metrics,
            'total_questions': len(data.get('questions', [])),
            'total_answers': len(data.get('answers', [])),
            'evaluation_timestamp': json.dumps(os.path.getmtime(__file__))
        }
        
        ## save json to ./persistent/eval_history/{filename.json}
        try:
            eval_dir = os.path.join(self.persistent_path, 'eval_history')
            os.makedirs(eval_dir, exist_ok=True)
            
            eval_path = os.path.join(eval_dir, f'{filename}.json')
            
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            result['saved_path'] = eval_path
            
        except Exception as e:
            result['save_error'] = str(e)
        
        return result
    
    def calculate_kraepelin_metrics(self, results, column_stats):
        """Calculate Panker, Janker, Hanker, and Tianker metrics"""
        columns = list(column_stats.keys())
        if len(columns) < 2:
            return {
                'panker': 0,
                'janker': 0,
                'hanker': 0,
                'tianker': 0,
                'error': 'Insufficient columns for metric calculation'
            }
        
        # Get performance data for each column
        column_performance = []
        for col in sorted(columns):
            column_performance.append(column_stats[col]['correct'])
        
        # Calculate metrics
        panker = self.calculate_panker(column_performance)
        janker = self.calculate_janker(column_performance)
        hanker = self.calculate_hanker(column_performance)
        tianker = self.calculate_tianker(column_performance)
        
        return {
            'panker': panker,
            'janker': janker,
            'hanker': hanker,
            'tianker': tianker,
            'column_performance': column_performance,
            'performance_trend': self.analyze_performance_trend(column_performance)
        }
    
    def calculate_panker(self, performance):
        """
        Panker: Measures work consistency
        Formula: (Highest Score - Lowest Score) / Average Score
        Lower values indicate better consistency
        """
        if len(performance) < 2:
            return 0
        
        max_score = max(performance)
        min_score = min(performance)
        avg_score = sum(performance) / len(performance)
        
        if avg_score == 0:
            return 0
        
        panker = (max_score - min_score) / avg_score
        return round(panker, 4)
    
    def calculate_janker(self, performance):
        """
        Janker: Measures work rhythm/tempo stability
        Formula: Sum of absolute differences between consecutive columns / (n-1)
        Lower values indicate better rhythm stability
        """
        if len(performance) < 2:
            return 0
        
        total_diff = 0
        for i in range(1, len(performance)):
            total_diff += abs(performance[i] - performance[i-1])
        
        janker = total_diff / (len(performance) - 1)
        return round(janker, 4)
    
    def calculate_hanker(self, performance):
        """
        Hanker: Measures work decline/fatigue
        Formula: (Performance in first half - Performance in second half) / Total performance
        Positive values indicate fatigue/decline
        """
        if len(performance) < 2:
            return 0
        
        mid_point = len(performance) // 2
        first_half = sum(performance[:mid_point])
        second_half = sum(performance[mid_point:])
        total_performance = sum(performance)
        
        if total_performance == 0:
            return 0
        
        hanker = (first_half - second_half) / total_performance
        return round(hanker, 4)
    
    def calculate_tianker(self, performance):
        """
        Tianker: Measures work acceleration/improvement
        Formula: (Performance in second half - Performance in first half) / Total performance
        Positive values indicate improvement/acceleration
        """
        if len(performance) < 2:
            return 0
        
        mid_point = len(performance) // 2
        first_half = sum(performance[:mid_point])
        second_half = sum(performance[mid_point:])
        total_performance = sum(performance)
        
        if total_performance == 0:
            return 0
        
        tianker = (second_half - first_half) / total_performance
        return round(tianker, 4)
    
    def analyze_performance_trend(self, performance):
        """Analyze overall performance trend"""
        if len(performance) < 2:
            return "insufficient_data"
        
        # Calculate trend using simple linear regression slope
        n = len(performance)
        x_mean = (n - 1) / 2
        y_mean = sum(performance) / n
        
        numerator = sum((i - x_mean) * (performance[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def serialize_results(self, results):
        """Convert results to JSON-serializable format"""
        serialized = {}
        for col in results:
            serialized[str(col)] = {}
            for row in results[col]:
                serialized[str(col)][str(row)] = {
                    'is_correct': results[col][row].is_correct,
                    'column': results[col][row].column,
                    'row': results[col][row].row
                }
        return serialized

def create_save_and_evaluate_blueprint(cfg):
    save_and_evaluate_handler = SaveAndEvaluateHandler(cfg=cfg)
    save_and_evaluate_bp = Blueprint('save_and_evaluate_controller', __name__)

    @save_and_evaluate_bp.route('/save_and_evaluate', methods=['POST'])
    def save_and_evaluate():
        try:
            # Get filename from request payload
            payload_data = request.get_json()
            if not payload_data:
                return jsonify({"error": "No JSON payload provided"}), 400
            
            filename = payload_data.get('filename')
            if not filename:
                return jsonify({"error": "Filename is required in payload"}), 400
            
            result = save_and_evaluate_handler.handle_save_and_evaluate(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return save_and_evaluate_bp