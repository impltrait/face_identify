import os

PR = os.path.dirname(__file__)

# 选择模型
model = 'hog'
# 不相似 阈值
threshold = 0.3

# 加载人脸检测器
cnn_face_detection_model_v1_path = os.path.join(PR, "weights/mmod_human_face_detector.dat")
cascadeClassifier_path = os.path.join(PR, 'weights/haarcascade_frontalface_default.xml')

# 加载关键点检测器
shape_predictor_path = os.path.join(PR, 'weights/shape_predictor_68_face_landmarks.dat')
# 加载resnet模型
face_recognition_model_v1_path = os.path.join(PR, 'weights/dlib_face_recognition_resnet_model_v1.dat')

# face特征存储
feature_path = os.path.join(PR, "data/feature.csv")

# face特征绘制保持
save_path = os.path.join(PR, "regiser_face_save/")
