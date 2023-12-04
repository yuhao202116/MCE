import numpy as np
from tensorflow.lite.python import interpreter as tflite_interpreter
import time
from tqdm import tqdm
from utils import getAllFilePath, readGraphFromFile

def evaluate_tflite_model(model_path, test_case_num=100):
    # Load the TFLite model and allocate tensors
    interpreter = tflite_interpreter.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get all file paths for the test dataset
    path_df = getAllFilePath()

    # Initialize counters
    false_count = 0
    start_time = time.perf_counter_ns()

    # Evaluate the model on the test dataset
    for filename in tqdm(path_df[0:test_case_num], total=test_case_num):
        input_data, tag = readGraphFromFile(filename)
        input_data = np.resize(input_data, (1, 175, 175, 1))
        input_data = input_data.astype(np.float32)  # Ensure correct dtype
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output_data[0]) != tag:
            false_count += 1

    # Calculate the elapsed time
    end_time = time.perf_counter_ns()
    time_cost = (end_time - start_time) / 1e9
    single_graph_time = (end_time - start_time) / (1e6 * test_case_num)
    accuracy = 100 * (test_case_num - false_count) / test_case_num

    # Return the results
    return test_case_num, time_cost, single_graph_time, accuracy

# Example usage:
if __name__ == "__main__":
    MODEL_PATH = './MCE_model.tflite'
    test_case_num, time_cost, single_graph_time, accuracy = evaluate_tflite_model(MODEL_PATH, test_case_num=100)
    print(f"{test_case_num} example time cost = {time_cost}s, single graph = {single_graph_time}ms acc = {accuracy}%")
