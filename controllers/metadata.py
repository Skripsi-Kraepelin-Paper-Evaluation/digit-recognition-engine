from flask import Blueprint, jsonify, request
import os
import json

class MetadataHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_metadata(self, filename):

        preview_dir = f'{self.persistent_path}/metadata'
        json_filename = f'{filename}.json'
        json_path = os.path.join(preview_dir, json_filename)
        
        # Check if the JSON file exists
        if not os.path.exists(json_path):
            return {
                'error': 'File not found',
                'message': f'Metadata for {filename} does not exist',
                'filename': filename,
                'path': json_path
            }
        
        try:
            ## open json from ./persistent/metadata/{filename.json} parse and return result
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



def create_metadata_blueprint(cfg):
    metadata_handler = MetadataHandler(persistent_path=cfg.persistent_path)
    metadata_bp = Blueprint('metadata_controller', __name__)

    @metadata_bp.route('/metadata/<filename>', methods=['GET'])
    def metadata(filename):
        try:
            result = metadata_handler.handle_metadata(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return metadata_bp