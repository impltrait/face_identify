# -*- coding: utf-8 -*-
import base64
import numpy as np
import cv2


def image2base64(image_path: str):
    """
    :desc 用户给出图片路径，将其转化为bash64形式
    :param image_path:
    :return:
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    base64_code = base64.b64encode(image_data).decode("utf-8")
    return base64_code


def base64_cv2(base64_code):
    """
    将base64编码解析成opencv可用图片
    base64_code: base64编码后数据
    Returns: cv2图像，numpy.ndarray
    """
    # base64解码
    base64_str = base64.b64decode(base64_code)

    # 转换为np数组
    np_array = np.frombuffer(base64_str, np.uint8)
    # 转换成opencv可用格式
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


def base64to_image(base64string: str, image_path: str):
    """
    decrib ：用户发过来图片的bash64形式字符串，函数读取后返回出去
    :param base64string:
    :param image_path:
    :return:imageData
    """
    image_data = base64.b64decode(base64string)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    return image_path
