# 基于云端图像处理的移动多节点实时人流量监控系统

## 文件说明

* **`documentation`包括文字报告和参考文献原文**
* `torch_gpu.yml`为conda环境
* flask框架下的网络后端，主程序`app.py`
  * 里面定义了各种行为
  * 以及更新yolo估计和ssdcn估计这两个数据的线程
* 前端文件html+js，在`templates`下面
* `predict.py`里有SSDCNet的推理测试
* `detect.py`里有YOLO的推理测试
* `inference`为素材文件夹
* `data`是数据集
* `camera.py`里面修改从哪里读取视频
  * line30
  * `camera = cv2.VideoCapture(Camera.video_source)`
  * 括号里是`0`则是webcam，是`rtsp字符串`则是IP摄像头
* `base_camera.py`里面有图像获取的线程

## 使用

* 根据`torch_gpu.yml`配好conda环境
* 执行`python3 app.py`即可开启服务器
* 参考demo视频
  * 链接: https://pan.baidu.com/s/1Xc3U5mapYqe6qp3jfm00jw  密码: jejg

## 轮子使用和贡献成员

* 其他的文件和文件夹是轮子必要的组件
  * https://github.com/ultralytics/yolov5.git
  * https://github.com/niveditarufus/PeopleCounter-SSDCNet.git
* original contributors
  * Daming Zhang
  * Muqi Li
  * Yueqian Liu