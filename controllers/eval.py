import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from flask import Blueprint, jsonify, request
import os
import json

class EvalHandler:
    def __init__(self, persistent_path='./persistent',host='http://localhost:5000'):
        self.persistent_path = persistent_path
        self.host

    def handle_eval(self):
        preview_dir = f'{self.persistent_path}/eval_history'

        try:
            payload = request.get_json()
            if not payload:
                raise ValueError("No JSON payload received")
            
            filename = payload.get('filename')
            if not filename:
                raise ValueError("Missing 'filename' in request")
                
            question_arrays = payload.get('questionArrays')
            if not question_arrays:
                raise ValueError("Missing 'questionArrays' in request")
                
            # Fixed: was getting 'questionArrays' twice instead of 'answerArrays'
            answer_arrays = payload.get('answerArrays')
            if not answer_arrays:
                raise ValueError("Missing 'answerArrays' in request")

            # Validate that arrays are properly structured
            if not isinstance(question_arrays, list) or not isinstance(answer_arrays, list):
                raise ValueError("questionArrays and answerArrays must be lists")

            analyzer = KraepelinAnalyzer(question_arrays=question_arrays, answer_arrays=answer_arrays,persistent_path=self.persistent_path)
            analyzer.plot_total_correct_answers(filename=filename)

            # Generate full report
            report = analyzer.generate_full_report()

            result = {
                "panker": f"{report['panker']:.3f}",
                "tianker": f"{report['tianker']}",
                "janker": f"{report['janker']}",
                "jankerv2": f"{report['jankerv2']:.3f}",  # Fixed: missing 'f' prefix
                "hanker": f"{report['hanker']['equation']}",
                "accuracy": f"{report['accuracy']}",
                "colScorePerMinute": f"{report['column_score_per_minute']:.3f}",
                "totalCorrectAns": f"{report['summary']['total_correct_answers']}",
                "plotImagePath":f"{self.host}/{filename}.pdf"
            }

            os.makedirs(preview_dir, exist_ok=True)
            
            # Save result as JSON file
            json_filename = f'{filename}.json'
            json_path = os.path.join(preview_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            return result
            
        except Exception as e:
            raise Exception(f"Error processing request: {str(e)}")

def create_eval_blueprint(cfg):
    eval_handler = EvalHandler(persistent_path=cfg.persistent_path,host=cfg.host)
    eval_bp = Blueprint('eval', __name__)

    @eval_bp.route('/eval', methods=['POST'])
    def eval():
        try:
            result = eval_handler.handle_eval()
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return eval_bp

class KraepelinAnalyzer:
    def __init__(self, question_arrays: List[List[int]], answer_arrays: List[List], persistent_path):
        """
        Initialize Kraepelin test analyzer

        Args:
            question_arrays: List of arrays containing questions for each column
            answer_arrays: List of arrays containing testee's answers for each column
        """
        self.question_arrays = question_arrays
        self.answer_arrays = answer_arrays
        self.processed_questions = self._process_questions()
        self.results = {}
        self.persistent_path = persistent_path

    def _process_questions(self) -> List[List[int]]:
        """
        Process question arrays by adding consecutive pairs
        Returns arrays with one fewer number than original
        """
        processed = []
        for question_array in self.question_arrays:
            if len(question_array) < 2:
                processed.append([])
                continue

            processed_column = []
            for i in range(len(question_array) - 1):
                # Ensure we're working with integers
                try:
                    num1 = int(question_array[i])
                    num2 = int(question_array[i + 1])
                    result = (num1 + num2) % 10
                    processed_column.append(result)
                except (ValueError, TypeError):
                    # Skip invalid entries
                    continue
            processed.append(processed_column)

        return processed

    def _compare_answers(self, column_idx: int) -> Dict:
        """
        Compare processed questions with testee answers for a specific column

        Returns:
            Dict containing correct, errors, and skipped counts
        """
        if column_idx >= len(self.processed_questions) or column_idx >= len(self.answer_arrays):
            return {"correct": 0, "errors": 0, "skipped": 0, "total_answered": 0, "not_answered": 0}

        processed_q = self.processed_questions[column_idx]
        answers = self.answer_arrays[column_idx]

        correct = 0
        errors = 0
        skipped = 0
        not_answered = 0

        # Only process up to the number of answers provided
        max_items = min(len(processed_q), len(answers))

        for i in range(max_items):
            answer = answers[i]
            expected = processed_q[i]

            # Handle different types of empty/null values
            if answer == '' or answer is None or str(answer).strip() == '':
                skipped += 1
            elif str(answer).lower() == 'x':
                errors += 1
            elif str(answer).lower() == 'n/a':
                not_answered += 1
            else:
                try:
                    # Convert answer to int for comparison
                    answer_int = int(answer)
                    if answer_int == expected:
                        correct += 1
                    else:
                        errors += 1
                except (ValueError, TypeError):
                    # If answer can't be converted to int, treat as error
                    errors += 1

        total_answered = max_items - skipped

        return {
            "correct": correct,
            "errors": errors,
            "skipped": skipped,
            "total_answered": total_answered,
            "not_answered": not_answered
        }

    def calculate_panker(self) -> float:
        """
        Panker: total correct answer divided by the total column
        """
        total_correct = 0
        total_columns = len(self.answer_arrays)

        for i in range(total_columns):
            comparison = self._compare_answers(i)
            total_correct += comparison["correct"]

        panker = total_correct / total_columns if total_columns > 0 else 0
        self.results["panker"] = panker
        return panker

    def calculate_tianker(self) -> int:
        """
        Tianker: errors answer plus skipped answer (according to arrays of answers)
        """
        total_errors = 0
        total_skipped = 0

        for i in range(len(self.answer_arrays)):
            comparison = self._compare_answers(i)
            total_errors += comparison["errors"]
            total_skipped += comparison["skipped"]

        tianker = total_errors + total_skipped
        self.results["tianker"] = tianker
        return tianker

    def calculate_janker(self) -> int:
        """
        Janker: highest score - lowest score (per column correct answers)
        """
        column_scores = []

        for i in range(len(self.answer_arrays)):
            comparison = self._compare_answers(i)
            column_scores.append(comparison["correct"])

        if not column_scores:
            janker = 0
        else:
            janker = max(column_scores) - min(column_scores)

        self.results["janker"] = janker
        self.results["column_scores"] = column_scores
        return janker

    def calculate_jankerv2(self) -> float:
        """
        Jankerv2: av.dev = Σfd/N (average deviation)
        """
        column_scores = []

        for i in range(len(self.answer_arrays)):
            comparison = self._compare_answers(i)
            column_scores.append(comparison["correct"])

        if not column_scores:
            jankerv2 = 0
        else:
            mean_score = sum(column_scores) / len(column_scores)
            total_deviation = sum(abs(score - mean_score) for score in column_scores)
            jankerv2 = total_deviation / len(column_scores)

        self.results["jankerv2"] = jankerv2
        return jankerv2

    def calculate_hanker(self) -> Dict:
        """
        Hanker: Linear regression y = a + bx
        where x is column number, y is correct answers per column
        """
        column_scores = []

        for i in range(len(self.answer_arrays)):
            comparison = self._compare_answers(i)
            column_scores.append(comparison["correct"])

        if len(column_scores) < 2:
            return {"a": 0, "b": 0, "equation": "y = 0 + 0x"}

        n = len(column_scores)
        x_values = list(range(1, n + 1))  # Column numbers 1, 2, 3, ...
        y_values = column_scores

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x_squared = sum(x * x for x in x_values)

        # Calculate b (slope)
        denominator = n * sum_x_squared - (sum_x ** 2)
        if denominator == 0:
            b = 0
        else:
            b = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate a (intercept)
        mean_y = sum_y / n
        mean_x = sum_x / n
        a = mean_y - b * mean_x

        hanker_result = {
            "a": a,
            "b": b,
            "equation": f"y = {a:.3f} + {b:.3f}x"
        }

        self.results["hanker"] = hanker_result
        return hanker_result

    def calculate_accuracy(self) -> int:
        """
        Accuracy: total errors answer in columns 6-10, 21-25, 36-40
        (according to arrays of answers, excluding blanks)
        """
        target_columns = []
        # Convert to 0-based indexing
        target_columns.extend(range(5, 10))   # columns 6-10
        target_columns.extend(range(20, 25))  # columns 21-25
        target_columns.extend(range(35, 40))  # columns 36-40

        total_errors = 0

        for col_idx in target_columns:
            if col_idx < len(self.answer_arrays):
                comparison = self._compare_answers(col_idx)
                total_errors += comparison["errors"]

        self.results["accuracy"] = total_errors
        return total_errors

    def calculate_column_score_per_minute(self) -> float:
        """
        Column score per minute: total correct answer divided by total column/2
        """
        total_correct = sum(self._compare_answers(i)["correct"]
                           for i in range(len(self.answer_arrays)))
        total_columns = len(self.answer_arrays)

        if total_columns == 0:
            score_per_minute = 0
        else:
            score_per_minute = total_correct / (total_columns / 2)

        self.results["column_score_per_minute"] = score_per_minute
        return score_per_minute

    def get_total_correct_per_column(self) -> List[int]:
        """
        Total correct answer per column
        """
        column_correct = []

        for i in range(len(self.answer_arrays)):
            comparison = self._compare_answers(i)
            column_correct.append(comparison["correct"])

        self.results["total_correct_per_column"] = column_correct
        return column_correct

    def column_by_column_analysis(self) -> List[Dict]:
        """
        Detailed column-by-column analysis
        """
        analysis = []

        for i in range(len(self.answer_arrays)):
            comparison = self._compare_answers(i)
            column_analysis = {
                "column": i + 1,
                "correct": comparison["correct"],
                "errors": comparison["errors"],
                "skipped": comparison["skipped"],
                "total_answered": comparison["total_answered"],
                "not_answered": comparison["not_answered"],
                "accuracy_rate": comparison["correct"] / comparison["total_answered"]
                                if comparison["total_answered"] > 0 else 0
            }
            analysis.append(column_analysis)

        self.results["column_analysis"] = analysis
        return analysis

    def plot_total_correct_answers(self, save_path: str = None, filename: str = None):
        """
        Create and save a graph of total correct answers per column
        
        Args:
            save_path: Directory path where to save the plot
            filename: Base filename (without extension) for the plot file
        
        Returns:
            str: Path to the saved plot file
        """
        column_correct = self.get_total_correct_per_column()
        column_numbers = list(range(1, len(column_correct) + 1))

        plt.figure(figsize=(12, 6))
        plt.plot(column_numbers, column_correct, marker='o', linewidth=2, markersize=4)
        plt.title('Total Correct Answers per Column - Kraepelin Test', fontsize=14, fontweight='bold')
        plt.xlabel('Column Number', fontsize=12)
        plt.ylabel('Correct Answers', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, len(column_correct) + 1, 5))

        # Add trend line (Hanker regression)
        hanker = self.calculate_hanker()
        if hanker["b"] != 0 or hanker["a"] != 0:
            trend_y = [hanker["a"] + hanker["b"] * x for x in column_numbers]
            plt.plot(column_numbers, trend_y, '--', color='red', alpha=0.7,
                    label=f'Trend: {hanker["equation"]}')
            plt.legend()

        plt.tight_layout()
        
        # Generate filename and save path
        if not filename:
            filename = "kraepelin_plot"
        
        if not save_path:
            save_path = f"{self.persistent_path}/eval_history/plots"
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Full file path with png extension
        file_path = os.path.join(save_path, f"{filename}.png")
        
        # Save the plot
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        return file_path

    def generate_full_report(self) -> Dict:
        """
        Generate complete analysis report
        """
        # Calculate all metrics
        panker = self.calculate_panker()
        tianker = self.calculate_tianker()
        janker = self.calculate_janker()
        jankerv2 = self.calculate_jankerv2()
        hanker = self.calculate_hanker()
        accuracy = self.calculate_accuracy()
        score_per_minute = self.calculate_column_score_per_minute()
        column_correct = self.get_total_correct_per_column()
        column_analysis = self.column_by_column_analysis()

        report = {
            "panker": panker,
            "tianker": tianker,
            "janker": janker,
            "jankerv2": jankerv2,
            "hanker": hanker,
            "accuracy": accuracy,
            "column_score_per_minute": score_per_minute,
            "total_correct_per_column": column_correct,
            "column_by_column_analysis": column_analysis,
            "summary": {
                "total_columns_analyzed": len(self.answer_arrays),
                "total_correct_answers": sum(column_correct),
                "average_correct_per_column": sum(column_correct) / len(column_correct) if column_correct else 0
            }
        }

        return report