from configs import config
from flask import Flask
from controllers import predict, preview_history, list_files_uploaded, save_and_evaluate, eval_history
from engines import inference


def create_app():
    app = Flask(__name__)

    cfg = config.AppConfig()

    # init inference engine
    inferencer = inference.NewDigitsRecogModel('./output_model/model0.h5',threshold_answer=cfg.threshold_answer,threshold_question=cfg.threshold_question)

    
    app.config.from_object(cfg)

    # Register blueprints
    app.register_blueprint(predict.create_predict_blueprint(inferencer,cfg))
    app.register_blueprint(preview_history.create_preview_history_blueprint(cfg))
    app.register_blueprint(list_files_uploaded.create_list_uploaded_blueprint(cfg))
    app.register_blueprint(save_and_evaluate.create_save_and_evaluate_blueprint(cfg))
    app.register_blueprint(eval_history.create_eval_history_blueprint(cfg))

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)