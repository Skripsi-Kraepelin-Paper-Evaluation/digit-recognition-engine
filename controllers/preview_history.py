from flask import Blueprint, jsonify, request
import os
import json
from database import get_db, PreviewHistory

class PreviewHistoryHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_preview_history(self, filename):
        db = get_db()
        try:
            # Try to get from database first
            preview = db.query(PreviewHistory).filter_by(filename=filename).first()
            
            if preview:
                result = {
                    'filename': preview.filename,
                    'questions': preview.questions,
                    'answers': preview.answers,
                    'total_questions': preview.total_questions,
                    'total_answers': preview.total_answers,
                    'loaded_from': 'database',
                    'loaded_at': preview.updated_at.timestamp() if preview.updated_at else None
                }
                return result
            
            # Fallback to file system if not in database
            preview_dir = f'{self.persistent_path}/preview_history'
            json_filename = f'{filename}.json'
            json_path = os.path.join(preview_dir, json_filename)
            
            if not os.path.exists(json_path):
                return {
                    'error': 'File not found',
                    'message': f'Preview history for {filename} does not exist',
                    'filename': filename,
                    'path': json_path
                }
            
            with open(json_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            result['loaded_from'] = json_path
            result['loaded_at'] = os.path.getmtime(json_path)
            
            return result
            
        except Exception as e:
            return {
                'error': 'Database error',
                'message': f'Failed to fetch preview history: {str(e)}',
                'filename': filename
            }
        finally:
            db.close()



def create_preview_history_blueprint(cfg):
    preview_history_handler = PreviewHistoryHandler(persistent_path=cfg.persistent_path)
    preview_history_bp = Blueprint('preview_history_controller', __name__)

    @preview_history_bp.route('/preview_history/<filename>', methods=['GET'])
    def preview_history(filename):
        try:
            result = preview_history_handler.handle_preview_history(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return preview_history_bp