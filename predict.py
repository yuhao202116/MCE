import time

import numpy as np
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

from utils import removeZeroPad


def predict(image_path, model_path):
    start_time = time.time()  # 记录开始时间
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 读取图像文件
    image = Image.open(image_path)
    image = image.convert("F")
    image = Image.fromarray(removeZeroPad(np.asarray(image)))
    image = image.resize((175, 175))
    image = np.asarray(image) / 255
    input_data = np.resize(image, (1, 175, 175, 1))
    input_data = input_data.astype(np.float32)

    # 设置输入数据
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 进行推理
    interpreter.invoke()

    # 获取输出结果
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = output_data[0]

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算花费时间
    return probabilities, elapsed_time


if __name__ == '__main__':
    MODEL_PATH = './MCE_model.tflite'
    image_path = "./5.png"
    pos, time = predict(image_path, MODEL_PATH)
