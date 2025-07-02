# main_module.py

from engines import inference
from models import predicted_digit_question as question

def run_prediction(image_path='./test_image/0.png', row=0, column=0):
    inferencer = inference.NewDigitsRecogModel('./output_model/model0.h5')

    try:
        digit, confidence, blank = inferencer.predict_digit(
            image_path,
            kernel_size=3,
            iterations=1,
        )

        if blank:
            question.PredictedDigitQuestion(is_blank=blank, column=column, row=row)
            return "blank", None, None

        need_manual_check = confidence < 80
        question.PredictedDigitQuestion(
            is_blank=blank,
            digit=digit,
            accuracy=confidence,
            row=row,
            column=column,
            need_manual_check=need_manual_check
        )
        return "digit", digit, confidence

    except FileNotFoundError:
        print("Image file not found.")
        return "error", None, None
