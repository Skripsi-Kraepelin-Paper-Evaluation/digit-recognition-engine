from configs import config
from flask import Flask
from controllers import predict, preview_history, list_files_uploaded, eval_history, upload_and_roi,eval
from engines import inference
from flask_cors import CORS



def create_app():
    cfg = config.AppConfig()
    app = Flask(__name__, static_url_path='/public/persistent',static_folder=cfg.persistent_path)
    CORS(app)


    # init inference engine
    inferencer = inference.NewDigitsRecogModel('./output_model/model0.h5',threshold_answer=cfg.threshold_answer)

    
    app.config.from_object(cfg)

    # Register blueprints
    app.register_blueprint(predict.create_predict_blueprint(inferencer,cfg))
    app.register_blueprint(preview_history.create_preview_history_blueprint(cfg))
    app.register_blueprint(list_files_uploaded.create_list_uploaded_blueprint(cfg))
    app.register_blueprint(eval_history.create_eval_history_blueprint(cfg))
    app.register_blueprint(upload_and_roi.create_upload_roi_blueprint(cfg))
    app.register_blueprint(eval.create_eval_blueprint(cfg))

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)