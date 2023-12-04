import numpy as np
import tflite_runtime.interpreter as tflite
import time
from utils import getAllFilePath, readGraphFromFile

MODEL_PATH = './MCE_model.tflite'
# MODEL_PATH = './mobilemodel.tflite'
# MODEL_PATH = './resnet50_model.tflite'


interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)
path_df = getAllFilePath()

test_case_num = 1000
start_time = time.perf_counter_ns()
false_count = 0
for filename in path_df[0:test_case_num]:
    input_data, tag = readGraphFromFile(filename)
    input_data = np.resize(input_data, (1, 175, 175, 1))
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data[0]) != tag:
        false_count += 1

end_time = time.perf_counter_ns()
print("{} example time cost = {}s, single graph = {}ms acc = {}%".format(test_case_num, (end_time - start_time) / 1e9,
                                                                         (end_time - start_time) / (
                                                                                     1e6 * test_case_num),
                                                                         100 - false_count))
