import json
from flask import Flask, request
import timeit
from log import logger
from src.model import FACE
from src.image2Bash64 import base64_cv2

import uuid

app = Flask(__name__)

start_time = timeit.default_timer()

face = FACE()

end_time = timeit.default_timer()
logger.info("模型初始化完成，用时为：%s", end_time - start_time)


def with2image_base64(image_base64string):
    """
    :param image_base64string:
    """
    base64_cv2_image = None
    try:
        # 加载图像 base64
        image_base64string = image_base64string.split(',')
        if len(image_base64string) > 1:
            image_base64string = image_base64string[1]
        else:
            image_base64string = image_base64string[0]
        base64_cv2_image = base64_cv2(image_base64string)
    except Exception as e:
        logger.info(f"图片base64 解失败！！！")
        raise ValueError(f"图片base64 解失败！！！\n{e}")
    finally:

        return base64_cv2_image


@app.route("/face/register", methods=['post'])
def after_registration():
    """
    人脸注册
    """
    ret = {}
    image_base64string = request.json.get('imageBase64String')  # 接收post参数
    user_name = request.json.get('userName')  # 接收post参数
    if not image_base64string:
        # 传入参数为空的错误控制
        ret['code'] = "98"
        ret['message'] = "缺少必要传入参数"
        logger.info(ret)
        return json.dumps(ret, ensure_ascii=True)

    image_base64string = with2image_base64(image_base64string)
    rel = False
    if image_base64string is not None:
        rel = face.register_face_information(image_base64string, face_id=uuid.uuid4(), user_name=user_name)

    if rel:
        ret['code'] = "00"
        ret['message'] = '人脸注册信息成功'
        logger.info(ret)
        return json.dumps(ret, ensure_ascii=True)
    else:
        ret['code'] = "98"
        ret['message'] = '人脸注册信息失败'
        logger.info(ret)
        return json.dumps(ret, ensure_ascii=True)


@app.route("/face/identify", methods=['post'])
def off_face_recognition():
    """
    人脸识别
    """
    ret = {}
    image_base64string = request.json.get('imageBase64String')  # 接收post参数
    if not image_base64string:
        # 传入参数为空的错误控制
        ret['code'] = "98"
        ret['message'] = "缺少必要传入参数"
        logger.info(ret)
        return json.dumps(ret, ensure_ascii=True)

    image_base64string = with2image_base64(image_base64string)
    rel = []
    if image_base64string is not None:
        rel = face.face_identify(image_base64string)
    ret['code'] = "00"
    ret['face'] = rel
    logger.info(ret)
    return json.dumps(ret, ensure_ascii=True)


if __name__ == '__main__':
    # flask初始化
    app.run(
        port=8871,
        debug=False,
        threaded=False,
        host='0.0.0.0'
    )
