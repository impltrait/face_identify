# face_identify
本项目基于dlib做人脸检测，跟数据库人脸做相似的计算

# 人脸注册接口
- http://192.168.11.107:8871/face/register
- POST请求
- 参数
```
{
    "imageBase64String":"", # 图片base64
    "userName":"test" # 注册者的姓名
}
```
# 人脸识别 相似用户
- http://192.168.11.107:8871/face/identify
- POST请求
- 参数
```
{
    "imageBase64String":"", # 图片base64
}
```
- 结果
```
{
    "code": "00",
    "face": [
        {
            "faceID": "82bc5c34-2ae4-47c1-8e73-1163ede0d1ec",
            "userName": "test"
        }
    ]
}
```
