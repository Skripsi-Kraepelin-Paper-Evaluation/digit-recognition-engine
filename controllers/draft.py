from flask import Blueprint, jsonify, request
import os
import json
import shutil

class DraftHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_save_draft(self, filename):
        #parsing payload

        #save to persistent based on filename

        return

    def handle_load_draft(self, filename):

        #load from persistent

        #parsing

        #send response

        return


def create_draft_blueprint(cfg):
    draft_handler = DraftHandler(persistent_path=cfg.persistent_path)
    draft_bp = Blueprint('draft_controller', __name__)

    #save as draft
    @draft_bp.route('/draft/<filename>', methods=['POST'])
    def draft(filename):
        try:
            result = draft_handler.handle_save_draft(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    #load draft
    @draft_bp.route('/draft/<filename>', methods=['GET'])
    def draft(filename):
        try:
            result = draft_handler.handle_load_draft(filename)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    
    return draft_bp