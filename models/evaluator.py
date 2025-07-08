from models import predicted_digit_answer as pa
from models import predicted_digit_question as pq


class CorrectResult:
    def __init__(self, is_correct=False,column=0,row=0):
        self.is_correct = is_correct
        self.column = column
        self.row = row

class Evaluator:
    ## initialize from questions top, questions bottom, answer top
    def __init__(self, row_count,col_count, questions, answers):
        self.row_count = row_count
        self.questions = questions
        self.answers = answers
        self.col_count = col_count
        if len(self.answers) != (len(self.questions)-row_count):
            print("size problem")
            exit()
        
        # Validate size: answers should be (total_questions - row_count) because each column has (row_count-1) answers
        expected_answers = col_count * (row_count - 1)
        if len(self.answers) != expected_answers:
            print(f"Size problem: Expected {expected_answers} answers, got {len(self.answers)}")
            exit()
    
    def get_question_by_position(self, column, row):
        """Get question at specific column and row"""
        for q in self.questions:
            if q.column == column and q.row == row:
                return q
        return None
    
    def get_answer_by_position(self, column, row):
        """Get answer at specific column and row"""
        for a in self.answers:
            if a.column == column and a.row == row:
                return a
        return None
    
    def validate_manual_check(self, item):
        """Validate if manual check requirements are met"""
        if item.need_manual_check and not item.checked:
            return False
        return True
    
    def evaluate(self):
        """Evaluate kraepelin raw test"""
        results = {}
        
        # Loop through each column
        for col in range(self.col_count):
            results[col] = {}
            
            # Loop through each row (except the last one, since we need pairs)
            for row in range(self.row_count - 1):
                # Get the two consecutive questions
                question1 = self.get_question_by_position(col, row)
                question2 = self.get_question_by_position(col, row + 1)
                
                # Get the corresponding answer
                answer = self.get_answer_by_position(col, row)
                
                # Initialize result as incorrect
                result = CorrectResult(is_correct=False, column=col, row=row)
                
                # Check if all required items exist
                if question1 is None or question2 is None or answer is None:
                    print(f"Missing data at column {col}, row {row}")
                    results[col][row] = result
                    continue
                
                # Check if any item is blank
                if question1.is_blank or question2.is_blank or answer.is_blank:
                    print(f"Blank item found at column {col}, row {row}")
                    results[col][row] = result
                    continue
                
                # Validate manual check requirements
                if not (self.validate_manual_check(question1) and 
                        self.validate_manual_check(question2) and 
                        self.validate_manual_check(answer)):
                    print(f"Manual check validation failed at column {col}, row {row}")
                    results[col][row] = result
                    continue
                
                # Calculate the sum and check if it matches the answer
                sum_questions = (question1.digit + question2.digit) % 10
                if sum_questions == answer.digit:
                    result.is_correct = True
                
                results[col][row] = result
        
        return results
    def get_statistics(self, results):
        """Get evaluation statistics"""
        total_evaluated = 0
        total_correct = 0
        
        for col in results:
            for row in results[col]:
                total_evaluated += 1
                if results[col][row].is_correct:
                    total_correct += 1
        
        accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
        
        return {
            'total_evaluated': total_evaluated,
            'total_correct': total_correct,
            'total_incorrect': total_evaluated - total_correct,
            'accuracy_percentage': round(accuracy, 2)
        }
    def get_column_statistics(self, results):
        """Get statistics per column"""
        column_stats = {}
        
        for col in results:
            correct = 0
            total = 0
            
            for row in results[col]:
                total += 1
                if results[col][row].is_correct:
                    correct += 1
            
            accuracy = (correct / total * 100) if total > 0 else 0
            column_stats[col] = {
                'total': total,
                'correct': correct,
                'incorrect': total - correct,
                'accuracy_percentage': round(accuracy, 2)
            }
        
        return column_stats



# Example usage function to parse from JSON-like data
def create_evaluator_from_json(data,cfg):
    """Create evaluator from JSON data structure"""
    questions = []
    answers = []
    
    # Parse questions
    for q_data in data['questions']:
        question = pq.PredictedDigitQuestion(
            digit=q_data['digit'],
            accuracy=q_data['accuracy'],
            column=q_data['column'],
            row=q_data['row'],
            need_manual_check=q_data['need_manual_check'],
            checked=q_data['checked'],
            is_blank=q_data['is_blank']
        )
        questions.append(question)
    
    # Parse answers
    for a_data in data['answers']:
        answer = pa.PredictedDigitAnswer(
            digit=a_data['digit'],
            accuracy=a_data['accuracy'],
            column=a_data['column'],
            row=a_data['row'],
            need_manual_check=a_data['need_manual_check'],
            checked=a_data['checked'],
            is_blank=a_data['is_blank']
        )
        answers.append(answer)
    
    return Evaluator(cfg.row, cfg.col, questions, answers)