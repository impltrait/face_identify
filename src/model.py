# -*- coding: utf-8 -*-
import cv2
import dlib
import csv
from settings import *
from log import logger
import numpy as np


class FACE:
    def __init__(self):
        self.model = model
        # 加载人脸检测器
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.cnn_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model_v1_path)
        self.haar_face_detector = cv2.CascadeClassifier(cascadeClassifier_path)

        # 加载关键点检测器
        self.points_detector = dlib.shape_predictor(shape_predictor_path)
        # 加载resnet模型
        self.face_descriptor_extractor = dlib.face_recognition_model_v1(face_recognition_model_v1_path)
        # 特征
        self.feature_list, self.label_list, self.name_list = None, [], []
        self.get_face_list()
        self.flag = True

    def get_dlib_rect(self, face):
        if self.model == 'hog':
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            return map(lambda x: x if x >= 0 else 0, (l, t, r, b))

        if self.model == 'cnn':
            l = face.rect.left()
            t = face.rect.top()
            r = face.rect.right()
            b = face.rect.bottom()
            return map(lambda x: x if x >= 0 else 0, (l, t, r, b))
        # 默认 haar
        l = face[0]
        t = face[1]
        r = face[0] + face[2]
        b = face[1] + face[3]
        return map(lambda x: x if x >= 0 else 0, (l, t, r, b))

    def face_detection_model(self, image_base64, ):
        # 切换人脸检测器
        if self.model == 'hog':
            face_detection = self.hog_face_detector(image_base64, 1)
            return face_detection

        if self.model == 'cnn':
            face_detection = self.cnn_detector(image_base64, 1)
            return face_detection
        # 默认 haar
        frame_gray = cv2.cvtColor(image_base64, cv2.COLOR_BGR2GRAY)
        face_detection = self.haar_face_detector.detectMultiScale(frame_gray, minNeighbors=7, minSize=(100, 100))
        return face_detection

    def get_face_list(self):
        logger.info('加载注册的人脸特征')
        self.feature_list, self.label_list, self.name_list = None, [], []
        # 加载保存的特征样本
        with open(feature_path, 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                # 重新加载数据
                face_id = line[0]
                user_name = line[1]
                face_descriptor = eval(line[2])

                self.label_list.append(face_id)
                self.name_list.append(user_name)

                # 转为numpy格式
                face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
                # 转为二维矩阵，拼接
                face_descriptor = np.reshape(face_descriptor, (1, -1))
                # 初始化
                if self.feature_list is None:
                    self.feature_list = face_descriptor
                else:
                    # 拼接
                    self.feature_list = np.concatenate((self.feature_list, face_descriptor), axis=0)
        logger.info("特征加载完毕")

    def register_face_information(self, image_base64, face_id=1, user_name='default'):
        # 打开文件
        f = open(feature_path, 'a', newline='')
        csv_writer = csv.writer(f)
        try:

            # 检测人脸
            face_detection = self.face_detection_model(image_base64)
            for face in face_detection:

                face_descriptor, points, l, t, r, b = self.get_face_descriptor(face, image_base64=image_base64)

                # 描述符增加进data文件
                line = [face_id, user_name, face_descriptor]
                # 写入
                csv_writer.writerow(line)

                # 保存照片样本
                # print('人脸注册成功 faceId:{faceId}，userName:{userName}'.format(faceId=face_id, userName=user_name))
                logger.info('人脸注册成功 faceId:{faceId}，userName:{userName}'.format(faceId=face_id, userName=user_name))

                # 绘制人脸关键点
                for point in points.parts():
                    cv2.circle(image_base64, (point.x, point.y), 2, (255, 0, 255), 1)
                cv2.rectangle(image_base64, (l, t), (r, b), (0, 255, 0), 2)
                # 保存
                cv2.imwrite(os.path.join(save_path, f"{user_name}@{face_id}.jpg"), image_base64)
                # cv2.imshow('Face Attendance Demo: Register', image_base64)
                # cv2.waitKey(0)
        except Exception as e:
            self.flag = False
            logger.info(f"{user_name} 人脸信息注册失败！！！")
            raise ValueError(f"{user_name} 人脸信息注册失败！！！\n{e}")
        finally:
            f.close()
            cv2.destroyAllWindows()

            if self.flag:
                # 注册完 重新更新 加载特征
                self.get_face_list()
                return self.flag
            else:
                self.flag = True
                return False

    def face_identify(self, image_base64):
        rel = []
        try:
            # 检测人脸
            face_detection = self.face_detection_model(image_base64)
            for face in face_detection:
                face_descriptor, _, _, _, _, _ = self.get_face_descriptor(face, image_base64=image_base64)
                face_descriptor = np.asarray(face_descriptor, dtype=np.float64)

                # 计算距离
                distance = np.linalg.norm((face_descriptor - self.feature_list), axis=1)
                # 最小距离索引
                min_index = np.argmin(distance)
                # 最小距离
                min_distance = distance[min_index]
                logger.info("不相似：{}".format(min_distance))
                if min_distance < threshold:
                    # 距离小于阈值，表示匹配
                    predict_id = self.label_list[min_index]
                    predict_name = self.name_list[min_index]
                    rel.append({"faceID": predict_id, "userName": predict_name})
        except Exception as e:
            logger.info(f"可能人脸检测信息！！！")
            raise ValueError(f"可能人脸检测信息！！！\n{e}")
        finally:
            return rel

    def get_face_descriptor(self, face, image_base64):
        l, t, r, b = self.get_dlib_rect(face)

        face = dlib.rectangle(l, t, r, b)

        # 识别68个关键点
        points = self.points_detector(image_base64, face)

        # 特征
        face_descriptor = self.face_descriptor_extractor.compute_face_descriptor(image_base64, points)
        return [f for f in face_descriptor], points, l, t, r, b
