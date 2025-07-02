# benchmark.py

import time
from main_module import run_prediction

def benchmark(iterations=3600, image_path='./test_image/0.png'):
    successes = 0
    blanks = 0
    errors = 0

    start_time = time.time()

    for i in range(iterations):
        result_type, digit, confidence = run_prediction(image_path=image_path)

        print(f'iteration no {i+1}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Time (s)        : {elapsed_time:.2f}")
    print(f"Average Time per Run  : {elapsed_time / iterations:.4f} seconds")

if __name__ == "__main__":
    benchmark()
