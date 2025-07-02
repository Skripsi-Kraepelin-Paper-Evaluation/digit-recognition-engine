from engines import inference
from models import predicted_digit_question as question

# Example usage
if __name__ == "__main__":
    # Initialize the enhanced inference class
    inferencer = inference.NewDigitsRecogModel('./output_model/model0.h5')

    
    try:
        image_path = './test_image/0.png'  # Change this to your image 
        
        row = 0
        column = 0

        digit, confidence, is_blank = inferencer.predict_digit(
            image_path,
            kernel_size=3,
            iterations=1,
        )

        if is_blank:
            newQuestions = question.PredictedDigitQuestion(is_blank=is_blank, column=column,row=row)
            return

            
        print(f"Is Blank: {is_blank}")
        print(f"Predicted digit: {digit}")
        print(f"Confidence: {confidence:.3f}")
        need_manual_check = False
        if confidence < 80:
            need_manual_check = True

        newQuestions = question.PredictedDigitQuestion(is_blank=is_blank,digit=digit,accuracy=confidence,row=row,column=column,need_manual_check=need_manual_check)


    except FileNotFoundError:
        print("Image file not found. Please provide a valid path to a PNG image.")