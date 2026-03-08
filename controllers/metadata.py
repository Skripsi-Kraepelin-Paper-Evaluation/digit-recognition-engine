from flask import Blueprint, jsonify, request
import os
import json
from database import get_db, KraepelinProject

class MetadataHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_metadata(self, filename):
        db = get_db()
        try:
            # Try to get from database first
            project = db.query(KraepelinProject).filter_by(filename=filename).first()
            
            if project:
                result = {
                    'name': project.name,
                    'occupacyAndRole': project.occupacy_and_role,
                    'lastEdu': project.last_edu,
                    'pob': project.pob,
                    'dob': project.dob,
                    'testDate': project.test_date,
                    'loaded_from': 'database',
                    'loaded_at': project.updated_at.timestamp() if project.updated_at else None
                }
                return result
            
            # Fallback to file system if not in database
            preview_dir = f'{self.persistent_path}/metadata'
            json_filename = f'{filename}.json'
            json_path = os.path.join(preview_dir, json_filename)
            
            if not os.path.exists(json_path):
                return {
                    'error': 'File not found',
                    'message': f'Metadata for {filename} does not exist',
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
                'message': f'Failed to fetch metadata: {str(e)}',
                'filename': filename
            }
        finally:
            db.close()



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