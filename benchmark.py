import os
import glob
import time
import re
import json
import multiprocessing

# Assuming these are imported from your project structure
from algorithm.cp import CP
from algorithm.mip import MIP
from algorithm.heuristic import LocalSearch

# ==========================================
# 1. IMPORT OR DEFINE YOUR FUNCTIONS HERE
# ==========================================
def my_input_parser(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        inp = f.readlines()

    N = int(inp[0])
    nodes = [i for i in range(N + 1)]
    e = [0]      
    l = [99999]  
    d = [0]      
    c = []

    for i in range(N):
        e_i, l_i, d_i = inp[i + 1].split()
        e.append(int(e_i))
        l.append(int(l_i))
        d.append(int(d_i))

    for i in nodes:
        c.append([int(x) for x in inp[i + N + 1].split()])

    l[0] = max(l[i] + d[i] + c[i][0] for i in range(1, N + 1))

    return N, e, l, d, c


# ==========================================
# 2. TIMEOUT & OOM WORKER LOGIC
# ==========================================
def _worker_process(queue, func, args):
    """Executes the algorithm and puts the result in a thread-safe queue."""
    try:
        result = func(*args)
        queue.put(("SUCCESS", result))
    except MemoryError:
        # Catch Python-level Out Of Memory errors
        queue.put(("OOM", "MemoryError"))
    except Exception as e:
        queue.put(("ERROR", e))


def run_with_timeout(func, args, timeout):
    """Runs a function in a separate process, checking for timeouts and OOM crashes."""
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_worker_process, args=(queue, func, args))
    
    start_time = time.perf_counter()
    process.start()
    
    # Wait for the process to finish, but only up to 'timeout' seconds
    process.join(timeout)
    
    if process.is_alive():
        # If it's still alive after the timeout, it's a TLE
        process.terminate()
        process.join()
        return "TLE", timeout
    
    end_time = time.perf_counter()
    runtime = end_time - start_time
    
    # Check if the process died without putting anything in the queue
    if queue.empty():
        # Exit code -9 (SIGKILL) on Unix usually means the OS OOM Killer destroyed the process.
        # Exit codes vary by OS, but an empty queue means a hard crash occurred.
        if process.exitcode == -9:
            return "OOM", runtime
        else:
            raise RuntimeError(f"Worker process crashed unexpectedly with exit code: {process.exitcode}")

    # Process finished gracefully, extract the data
    status, payload = queue.get()
    
    if status == "SUCCESS":
        return payload, runtime
    elif status == "OOM":
        return "OOM", runtime
    else:
        raise payload  # Raise standard exceptions (e.g., ValueError, TypeError) in the main thread


# ==========================================
# 3. BENCHMARKING LOGIC
# ==========================================
def run_benchmark(folder_path, parser_func, target_func, time_limit=120):
    function_name = target_func.__name__
    results = []

    search_pattern = os.path.join(folder_path, '*.inp')
    inp_files = glob.glob(search_pattern)

    if not inp_files:
        print(f"No '.inp' files found in directory: '{folder_path}'")
        return

    def extract_file_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'n(\d+)\.inp', filename)
        return int(match.group(1)) if match else float('inf')

    inp_files.sort(key=extract_file_number)

    print(f"Found {len(inp_files)} test case(s). Starting benchmark...\n")

    for filepath in inp_files:
        filename = os.path.basename(filepath)
        match = re.search(r'n(\d+)\.inp', filename)
        n_value = int(match.group(1)) if match else filename 
        
        try:
            # Parse the inputs
            N, e, l, d, c = parser_func(filepath)
            
            # Run the algorithm with the enforced timeout
            result, runtime = run_with_timeout(target_func, (N, e, l, d, c), time_limit)
            
            if result == "TLE":
                cost = "TLE"
                recorded_runtime = ">120"
                print(f"Processed N={n_value}: TLE (Exceeded {time_limit}s)")
            elif result == "OOM":
                cost = "OOM"
                recorded_runtime = "OOM"
                print(f"Processed N={n_value}: OOM (Out Of Memory)")
            else:
                route, cost = result
                recorded_runtime = runtime
                print(f"Processed N={n_value}: Time = {runtime:.6f}s, Cost = {cost}")
            
            result_data = {
                "N": n_value,
                "runtime": recorded_runtime,
                "cost": cost
            }
            results.append(result_data)
            
        except Exception as err:
            print(f"Error processing N='{n_value}' ({filename}): {err}")

    # ==========================================
    # 4. DUMP RESULTS TO JSON
    # ==========================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    output_filename = f"{function_name}_results.json"
    output_path = os.path.join(results_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(results, out_file, indent=4)

    print(f"\nBenchmarking complete. JSON dumped to '{output_path}'.")


if __name__ == '__main__':
    TARGET_FOLDER = 'tests'  
    run_benchmark(TARGET_FOLDER, my_input_parser, LocalSearch, time_limit=120)