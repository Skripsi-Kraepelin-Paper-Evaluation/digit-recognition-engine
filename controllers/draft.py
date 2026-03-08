from flask import Blueprint, jsonify, request
import os
import json
from database import get_db
from database.models import Base
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from datetime import datetime

class DraftHistory(Base):
    __tablename__ = 'draft_history'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    draft_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DraftHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_save_draft(self, filename):
        payload = request.get_json()
        if not payload:
            return {'error': 'No data provided'}, 400
        
        db = get_db()
        try:
            # Ensure table exists
            from database.connection import get_database_url
            from sqlalchemy import create_engine
            engine = create_engine(get_database_url())
            DraftHistory.__table__.create(engine, checkfirst=True)
            
            draft = db.query(DraftHistory).filter_by(filename=filename).first()
            if draft:
                draft.draft_data = payload
            else:
                draft = DraftHistory(filename=filename, draft_data=payload)
                db.add(draft)
            db.commit()
            
            # Also save to file system as backup
            draft_dir = f'{self.persistent_path}/draft_history'
            os.makedirs(draft_dir, exist_ok=True)
            with open(f'{draft_dir}/{filename}.json', 'w') as f:
                json.dump(payload, f, indent=2)
            
            return {'success': True, 'message': 'Draft saved successfully'}
        except Exception as e:
            db.rollback()
            return {'error': f'Failed to save draft: {str(e)}'}, 500
        finally:
            db.close()

    def handle_load_draft(self, filename):
        db = get_db()
        try:
            # Ensure table exists
            from database.connection import get_database_url
            from sqlalchemy import create_engine
            engine = create_engine(get_database_url())
            DraftHistory.__table__.create(engine, checkfirst=True)
            
            draft = db.query(DraftHistory).filter_by(filename=filename).first()
            if draft:
                return {
                    'success': True,
                    'data': draft.draft_data,
                    'loaded_from': 'database',
                    'loaded_at': draft.updated_at.timestamp() if draft.updated_at else None
                }
            
            # Fallback to file system
            draft_path = f'{self.persistent_path}/draft_history/{filename}.json'
            if os.path.exists(draft_path):
                with open(draft_path, 'r') as f:
                    data = json.load(f)
                return {
                    'success': True,
                    'data': data,
                    'loaded_from': 'filesystem',
                    'loaded_at': os.path.getmtime(draft_path)
                }
            
            return {'error': 'Draft not found'}, 404
        except Exception as e:
            return {'error': f'Failed to load draft: {str(e)}'}, 500
        finally:
            db.close()


def create_draft_blueprint(cfg):
    draft_handler = DraftHandler(persistent_path=cfg.persistent_path)
    draft_bp = Blueprint('draft_controller', __name__)

    @draft_bp.route('/draft/<filename>', methods=['POST'])
    def save_draft(filename):
        result = draft_handler.handle_save_draft(filename)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result), 200

    @draft_bp.route('/draft/<filename>', methods=['GET'])
    def load_draft(filename):
        result = draft_handler.handle_load_draft(filename)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result), 200

    return draft_bp
