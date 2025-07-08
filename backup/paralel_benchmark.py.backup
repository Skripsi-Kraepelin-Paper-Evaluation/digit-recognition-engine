# parallel_benchmark.py

import time
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from main_module import run_prediction
import os

def calculate_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative estimates for CNN memory usage
    # Adjust these based on your specific model size
    estimated_model_memory_gb = 0.5  # Typical small CNN model
    max_memory_workers = int(memory_gb * 0.7 / estimated_model_memory_gb)  # Use 70% of RAM
    
    # CPU-based calculation
    cpu_workers = max(1, cpu_count - 1)  # Leave one core free
    
    # Take the minimum to avoid resource exhaustion
    optimal_workers = min(cpu_workers, max_memory_workers, 8)  # Cap at 8 for stability
    
    print(f"System specs:")
    print(f"  CPU cores: {cpu_count}")
    print(f"  RAM: {memory_gb:.1f} GB")
    print(f"  Calculated optimal workers: {optimal_workers}")
    
    return optimal_workers

def worker_task(args):
    """Worker function for parallel processing"""
    iteration_num, image_path = args
    try:
        start_time = time.time()
        result_type, digit, confidence = run_prediction(image_path=image_path)
        end_time = time.time()
        
        print(f'Iteration {iteration_num} completed in {end_time - start_time:.4f}s')
        return {
            'iteration': iteration_num,
            'result_type': result_type,
            'digit': digit,
            'confidence': confidence,
            'processing_time': end_time - start_time,
            'success': True
        }
    except Exception as e:
        print(f'Iteration {iteration_num} failed: {str(e)}')
        return {
            'iteration': iteration_num,
            'success': False,
            'error': str(e)
        }

def parallel_benchmark_process(iterations=3600, image_path='./test_image/0.png', num_workers=None):
    """Process-based parallelization (better for CPU-bound CNN inference)"""
    if num_workers is None:
        num_workers = calculate_optimal_workers()
    
    print(f"\nRunning benchmark with {num_workers} processes...")
    
    # Prepare tasks
    tasks = [(i+1, image_path) for i in range(iterations)]
    
    successes = 0
    errors = 0
    total_processing_time = 0
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(worker_task, task): task for task in tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            result = future.result()
            
            if result['success']:
                successes += 1
                total_processing_time += result['processing_time']
            else:
                errors += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate statistics
    avg_processing_time = total_processing_time / successes if successes > 0 else 0
    throughput = iterations / elapsed_time
    
    print(f"\n=== BENCHMARK RESULTS ===")
    print(f"Total iterations      : {iterations}")
    print(f"Successful runs       : {successes}")
    print(f"Failed runs           : {errors}")
    print(f"Workers used          : {num_workers}")
    print(f"Total wall time       : {elapsed_time:.2f} seconds")
    print(f"Average processing time: {avg_processing_time:.4f} seconds")
    print(f"Throughput            : {throughput:.2f} predictions/second")
    print(f"Speedup factor        : {throughput * avg_processing_time:.2f}x")

def parallel_benchmark_thread(iterations=3600, image_path='./test_image/0.png', num_workers=None):
    """Thread-based parallelization (use if your CNN releases GIL)"""
    if num_workers is None:
        num_workers = min(calculate_optimal_workers(), 4)  # Threads are limited by GIL
    
    print(f"\nRunning benchmark with {num_workers} threads...")
    
    tasks = [(i+1, image_path) for i in range(iterations)]
    
    successes = 0
    errors = 0
    total_processing_time = 0
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for I/O-bound or GIL-releasing tasks
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(worker_task, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            result = future.result()
            
            if result['success']:
                successes += 1
                total_processing_time += result['processing_time']
            else:
                errors += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    avg_processing_time = total_processing_time / successes if successes > 0 else 0
    throughput = iterations / elapsed_time
    
    print(f"\n=== THREAD BENCHMARK RESULTS ===")
    print(f"Total iterations      : {iterations}")
    print(f"Successful runs       : {successes}")
    print(f"Failed runs           : {errors}")
    print(f"Workers used          : {num_workers}")
    print(f"Total wall time       : {elapsed_time:.2f} seconds")
    print(f"Average processing time: {avg_processing_time:.4f} seconds")
    print(f"Throughput            : {throughput:.2f} predictions/second")

def benchmark_comparison(iterations=360, image_path='./test_image/0.png'):
    """Compare serial vs parallel performance"""
    print("=== PERFORMANCE COMPARISON ===")
    
    # Serial benchmark
    print("\n1. Serial processing...")
    start_time = time.time()
    for i in range(iterations):
        run_prediction(image_path=image_path)
        if (i + 1) % 50 == 0:
            print(f"Serial: {i+1}/{iterations} completed")
    
    serial_time = time.time() - start_time
    serial_throughput = iterations / serial_time
    
    print(f"Serial time: {serial_time:.2f}s, Throughput: {serial_throughput:.2f} pred/s")
    
    # Parallel benchmark
    print("\n2. Parallel processing...")
    parallel_benchmark_process(iterations, image_path)

if __name__ == "__main__":
    parallel_benchmark_process(3600)