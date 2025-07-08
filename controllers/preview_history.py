from flask import Blueprint, jsonify, request
import os
import json

class PreviewHistoryHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_preview_history(self, filename):

        preview_dir = f'{self.persistent_path}/preview_history'
        json_filename = f'{filename}.json'
        json_path = os.path.join(preview_dir, json_filename)
        
        # Check if the JSON file exists
        if not os.path.exists(json_path):
            return {
                'error': 'File not found',
                'message': f'Preview history for {filename} does not exist',
                'filename': filename,
                'path': json_path
            }
        
        try:
            ## open json from ./persistent/preview_history/{filename.json} parse and return result
            with open(json_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # Add metadata about the loaded file
            result['loaded_from'] = json_path
            result['loaded_at'] = os.path.getmtime(json_path)  # Last modified timestamp
            
        except json.JSONDecodeError as e:
            return {
                'error': 'Invalid JSON format',
                'message': f'Failed to parse JSON file: {str(e)}',
                'filename': filename,
                'path': json_path
            }
        except Exception as e:
            return {
                'error': 'File read error',
                'message': f'Failed to read file: {str(e)}',
                'filename': filename,
                'path': json_path
            }
        
        #####################################

        return result



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