from configs import config
from flask import Flask
from controllers import evaluate
from engines import inference


def create_app():
    app = Flask(__name__)

    # init inference engine
    inferencer = inference.NewDigitsRecogModel('./output_model/model0.h5')

    cfg = config.AppConfig()
    app.config.from_object(cfg)

    # Register blueprints
    app.register_blueprint(evaluate.create_eval_blueprint(inferencer))

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)