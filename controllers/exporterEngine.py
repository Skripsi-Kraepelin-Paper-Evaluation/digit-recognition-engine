from flask import Blueprint, jsonify, request
import os
import json
import shutil

class ExportHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_export(self, filename):
        # copy template
        # parsing payload
        # edit template
        # save to persistent
        # return result as byte stream to be downloaded



def create_export_blueprint(cfg):
    export_handler = ExportHandler(persistent_path=cfg.persistent_path)
    export_bp = Blueprint('export_controller', __name__)

    @export_bp.route('/export/<filename>', methods=['POST'])
    def export(filename):
        try:
            result = export_handler.handle_export(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    return export_bp
    