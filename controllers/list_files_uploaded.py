from flask import Blueprint, jsonify, request
import os
import json

class ListUploadedHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_list_uploaded(self):

        uploaded_dir = f'{self.persistent_path}/uploaded'
    
        # Check if the uploaded directory exists
        if not os.path.exists(uploaded_dir):
            return {
                'error': 'Directory not found',
                'message': 'Uploaded directory does not exist',
                'path': uploaded_dir,
                'files': []
            }
        
        try:
            # Get all files in the uploaded directory
            all_files = os.listdir(uploaded_dir)
            
            # Filter out directories, keep only files
            files_only = [f for f in all_files if os.path.isfile(os.path.join(uploaded_dir, f))]
            
            ## takeout the extensions
            files_without_extensions = []
            
            for file in files_only:
                # Get filename without extension
                filename_no_ext = os.path.splitext(file)[0]
                extension = os.path.splitext(file)[1]
                
                files_without_extensions.append(filename_no_ext)
            
            ## response
            response = {
                'success': True,
                'total_files': len(files_only),
                'files': files_without_extensions
            }
            
        except Exception as e:
            response = {
                'error': 'Failed to list files',
                'message': str(e),
                'path': uploaded_dir,
                'files': []
            }
        
        return response


def create_list_uploaded_blueprint(cfg):
    list_uploaded_handler = ListUploadedHandler(persistent_path=cfg.persistent_path)
    list_uploaded_bp = Blueprint('list_uploaded_controller', __name__)

    @list_uploaded_bp.route('/list_uploaded', methods=['GET'])
    def list_uploaded():
        try:
            result = list_uploaded_handler.handle_list_uploaded()
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return list_uploaded_bp