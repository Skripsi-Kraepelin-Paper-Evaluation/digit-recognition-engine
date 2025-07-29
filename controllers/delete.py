from flask import Blueprint, jsonify, request
import os
import json
import shutil

class DeleteHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_delete(self, filename):

        deleted_files = []
        errors = []
        
        # List of files/directories to delete
        targets = [
            f"{self.persistent_path}/roi_result/metadata/{filename}.json",
            f"{self.persistent_path}/roi_result/{filename}",
            f"{self.persistent_path}/eval_history/{filename}.json",
            f"{self.persistent_path}/eval_history/plots/{filename}.png",
            f"{self.persistent_path}/preview_history/{filename}.json",
            f"{self.persistent_path}/metadata/{filename}.json",
            f"{self.persistent_path}/uploaded/{filename}.pdf"
        ]
        
        for target in targets:
            try:
                if os.path.exists(target):
                    if os.path.isfile(target):
                        os.remove(target)
                        deleted_files.append(target)
                    elif os.path.isdir(target):
                        shutil.rmtree(target)
                        deleted_files.append(target)
                else:
                    # File/directory doesn't exist, but that's okay
                    pass
            except Exception as e:
                errors.append(f"Failed to delete {target}: {str(e)}")
        
        # Return result
        result = {
            "message": "Delete operation completed",
            "deleted_files": deleted_files,
            "deleted_count": len(deleted_files)
        }
        
        if errors:
            result["errors"] = errors
            result["error_count"] = len(errors)
        
        return result



def create_delete_blueprint(cfg):
    delete_handler = DeleteHandler(persistent_path=cfg.persistent_path)
    delete_bp = Blueprint('delete_controller', __name__)

    @delete_bp.route('/<filename>', methods=['DELETE'])
    def delete(filename):
        try:
            result = delete_handler.handle_delete(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return delete_bp