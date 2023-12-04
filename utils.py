import cv2
import numpy as np
import os
import random
from PIL import Image

labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
label_dict = {}
for i in range(4):
    label_dict[labels[i]] = i


def getAllFilePath():
    path_df = []
    a = ""
    for dirname, _, filenames in os.walk('./input'):
        a = dirname
        for filename in filenames:
            if 'images' in dirname:
                path_df.append(os.path.join(dirname, filename))
    random.shuffle(path_df)
    return path_df


def removeZeroPad(image):
    dummy = np.argwhere(image != 0)
    max_y = dummy[:, 0].max()
    min_y = dummy[:, 0].min()
    max_x = dummy[:, 1].max()
    min_x = dummy[:, 1].min()
    crop_image = image[min_y:max_y, min_x:max_x]
    return crop_image


def readGraphFromFile(path: str) -> (np.array, str):
    image = Image.open(path)
    image = image.convert("F")
    image = Image.fromarray(removeZeroPad(np.asarray(image)))
    image = image.resize((175, 175))
    # image.show()
    # 使用OpenCV显示图像
    # cv2.imshow('image', np.asarray(image))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return np.asarray(image) / 255, label_dict[path.split('\\')[-3]]
