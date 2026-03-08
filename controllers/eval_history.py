from flask import Blueprint, jsonify, request
import os
import json
from urllib.parse import urlparse, urlunparse
from database import get_db, EvalHistory

class EvalHistoryHandler:
    def __init__(self, cfg):
        self.persistent_path = cfg.persistent_path
        self.cfg = cfg

    def handle_eval_history(self, filename):
        db = get_db()
        try:
            # Try to get from database first
            eval_hist = db.query(EvalHistory).filter_by(filename=filename).first()
            
            if eval_hist:
                result = {
                    'panker': eval_hist.panker,
                    'tianker': eval_hist.tianker,
                    'janker': eval_hist.janker,
                    'jankerv2': eval_hist.jankerv2,
                    'hanker': eval_hist.hanker,
                    'accuracy': eval_hist.accuracy,
                    'colScorePerMinute': eval_hist.col_score_per_minute,
                    'totalCorrectAns': eval_hist.total_correct_ans,
                    'plotImagePath': eval_hist.plot_image_path,
                    'loaded_from': 'database',
                    'loaded_at': eval_hist.updated_at.timestamp() if eval_hist.updated_at else None
                }
                
                # Update plotImagePath host
                if result['plotImagePath']:
                    parsed_url = urlparse(result['plotImagePath'])
                    new_host = urlparse(self.cfg.host)
                    updated_url = parsed_url._replace(netloc=new_host.netloc, scheme=new_host.scheme)
                    result['plotImagePath'] = urlunparse(updated_url)
                
                return result
            
            # Fallback to file system if not in database
            preview_dir = f'{self.persistent_path}/eval_history'
            json_filename = f'{filename}.json'
            json_path = os.path.join(preview_dir, json_filename)
            
            if not os.path.exists(json_path):
                return {
                    'error': 'File not found',
                    'message': f'Eval history for {filename} does not exist',
                    'filename': filename,
                    'path': json_path
                }
            
            with open(json_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # Update plotImagePath host if it exists
            if 'plotImagePath' in result:
                parsed_url = urlparse(result['plotImagePath'])
                new_host = urlparse(self.cfg.host)
                updated_url = parsed_url._replace(netloc=new_host.netloc, scheme=new_host.scheme)
                result['plotImagePath'] = urlunparse(updated_url)
            
            result['loaded_from'] = json_path
            result['loaded_at'] = os.path.getmtime(json_path)
            
            return result
            
        except Exception as e:
            return {
                'error': 'Database error',
                'message': f'Failed to fetch eval history: {str(e)}',
                'filename': filename
            }
        finally:
            db.close()



def create_eval_history_blueprint(cfg):
    eval_history_handler = EvalHistoryHandler(cfg)
    eval_history_bp = Blueprint('eval_history_controller', __name__)

    @eval_history_bp.route('/eval_history/<filename>', methods=['GET'])
    def eval_history(filename):
        try:
            result = eval_history_handler.handle_eval_history(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return eval_history_bp