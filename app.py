#!/usr/bin/env python
import argparse
from importlib import import_module
import os
from flask import Flask, render_template, session, request, Response, redirect, url_for
import time

# import camera driver
from camera import Camera
import camera

# socket support
from flask_socketio import SocketIO, Namespace, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import threading
import logging
import socket

import pygame

# ssdcnet
import predict
import cv2

# yolo
import detect

optSource = None
app = Flask(__name__)
thread1 = None
thread2 = None
thread3 = None
thread4 = None
thread5 = None
thread6 = None

latitude = 0
longitude = 0
yolo_count = 0
ssdc_count = 0

bluetooth_ready = False

mutex = threading.Lock()

socketio = SocketIO(app, async_mode=None)

app.logger.disabled = True
log = logging.getLogger('werkzeug')
log.disabled = True

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def yolo_process_thread():
    """process video and return result"""
    global yolo_count
    yolov5s = detect.init_yolo()
    while not camera.stream_started:
        time.sleep(1)
    while True:
        camera.boxes, yolo_count = detect.detect_frame(camera.imgSrc, yolov5s, conf=opt.conf_thres, iou=opt.iou)

def yolo_emit_thread():
    while not camera.stream_started:
        time.sleep(1)
    while True:
        socketio.emit('my_response',
                      {'data': yolo_count},
                      namespace='/test')
        time.sleep(1)

def loc_emit_thread():
    while not camera.stream_started:
        time.sleep(1)
    while True:
        socketio.emit('loc',
                      {'lat': latitude,
                      'lon':longitude},
                      namespace='/test')
        time.sleep(1)

def ssdc_process_emit_thread():
    """init ssdcnet and update crowd estimation"""
    global ssdc_count
    enable_cuda = opt.cuda
    ssdc = predict.init_ssdc(cuda=enable_cuda)
    while not camera.stream_started:
        time.sleep(1)
    while True:
        ssdc_count = predict.detect_crowd(
            img=camera.imgSrc, net=ssdc, cuda=enable_cuda)
        socketio.emit('my_response_',
                      {'data': ssdc_count},
                      namespace='/test')

def getImg_thread():
    while not bluetooth_ready:
        time.sleep(1)
    video = cv2.VideoCapture(optSource)
    while True:
        _, camera.imgSrc = video.read()
        # read current frame
        if camera.stream_started == False:
            camera.stream_started = True

def bluetooth_thread():
    global latitude
    global longitude
    global bluetooth_ready
    serversocket = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
    host = socket.gethostname()  # 获取本地主机名
    port = 9999
    serversocket.bind((host, port))  # 绑定端口号
    serversocket.listen(5)  # 设置最大连接数，超过后排队
    myname = socket.getfqdn(socket.gethostname())
    myaddr = socket.gethostbyname(myname)
    print("My name", myname, "\r\nMy IP:", myaddr, "\r\nMy port:", port)
    car_ctrl = False
    car_ctrl_msg = "hello"
    moving_directionx = 0
    moving_directiony = 0
    while True:
        clientsocket, addr = serversocket.accept()  # 建立客户端连接
        clientsocket.setblocking(0)             # 把接受数据的方式设置为非阻塞的
        print("连接地址: %s" % str(addr))
        car_ctrl = True

        msg = 'BlueToothCar Server Connected' + "\r\n"
        clientsocket.send(msg.encode('utf-8'))

        car_index = input("输入所控制小车的序号: ")
        print("Car Index:", car_index, "\r\n", "请关闭小车控制的窗口再连接下一台蓝牙小车", "\r\n")

        pygame.init()
        screen = pygame.display.set_mode([640, 480])
        pygame.display.set_caption("蓝牙小车移动窗口")
        bg_color = [255, 255, 255]
        image = pygame.image.load("./车辆俯视图.png")

        image_start = pygame.image.load("./start.jpg") # TODO

        rect = image.get_rect()
        screen_rect = screen.get_rect()
        rect.center = screen_rect.center
        
        screen.fill([0, 0, 0])
        # screen.blit(image_start, rect)
        screen.blit(image_start, screen_rect.center)
        pygame.display.flip()

        continue_ = False
        while not continue_:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        continue_ = True

        if not bluetooth_ready:
            bluetooth_ready = True


        while car_ctrl:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    car_ctrl = False
                    car_ctrl_msg = "zPPm"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        moving_directiony = -1
                        car_ctrl_msg = "zW1m"
                    elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        moving_directionx = -1
                        car_ctrl_msg = "zA1m"
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        moving_directiony = 1
                        car_ctrl_msg = "zS1m"
                    elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        moving_directionx = 1
                        car_ctrl_msg = "zD1m"
                    else:
                        print("按键不合法，请重新输入")
                elif event.type == pygame.KEYUP:
                    moving_directionx = 0
                    moving_directiony = 0
                    car_ctrl_msg = "zPPm"

            time.sleep(0.05)
            rect.centerx = screen_rect.centerx + moving_directionx * 30
            rect.centery = screen_rect.centery + moving_directiony * 30
            screen.fill(bg_color)
            screen.blit(image, rect)
            pygame.display.flip()
            # 检查一下是否断开了连接再发送
            try:
                clientsocket.send(car_ctrl_msg.encode('utf-8'))
            except(ConnectionAbortedError, ConnectionAbortedError, ConnectionResetError, TypeError):
                print("连接已断开", "\r\n")
                break
            # 接收手机端传来的数据
            try:
                msg_recv = clientsocket.recv(1024)
                if not msg_recv:
                    continue
                elif msg_recv:  # 有数据的时候
                    msg_recv_str = msg_recv.decode('utf-8')
                    if msg_recv_str.startswith('LAT:'):
                        try:
                            latitude = float(msg_recv_str[4:])  # 纬度
                        except(Exception):  # 有时候socket数据阻塞导致要处理一串“39.12323LONG116.1888”会出现错误
                            continue
                    elif msg_recv_str.startswith('LONG:'):
                        try:
                            longitude = float(msg_recv_str[5:])  # 经度
                        except(Exception):
                            continue
                    else:
                        print("传来的数据：", msg_recv_str, "\r\n")
            except(ConnectionAbortedError, ConnectionAbortedError, ConnectionResetError, TypeError, BlockingIOError):
                continue
        pygame.quit()
        clientsocket.close()



@app.route('/')
def mapshow():
    """the people shows on the map page."""
    return render_template('mapshow.html', async_mode=socketio.async_mode)
    # return render_template('mapshow.html')

@app.route('/video')
def index():
    """Video streaming home page."""
    return render_template('index.html', async_mode=socketio.async_mode)
    # return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control')
def control():
    """camera fpv control page"""
    src = opt.source
    if opt.source == '0':
        src = 'video_feed'
    return redirect(src)


class MyNamespace(Namespace):
    def on_my_event(self, message):
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response',
             {'data': message['data'], 'count': session['receive_count']})

    def on_my_broadcast_event(self, message):
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response',
             {'data': message['data'], 'count': session['receive_count']},
             broadcast=True)

    def on_join(self, message):
        join_room(message['room'])
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response',
             {'data': 'In rooms: ' + ', '.join(rooms()),
              'count': session['receive_count']})

    def on_leave(self, message):
        leave_room(message['room'])
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response',
             {'data': 'In rooms: ' + ', '.join(rooms()),
              'count': session['receive_count']})

    def on_close_room(self, message):
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response', {'data': 'Room ' + message['room'] + ' is closing.',
                             'count': session['receive_count']},
             room=message['room'])
        close_room(message['room'])

    def on_my_room_event(self, message):
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response',
             {'data': message['data'], 'count': session['receive_count']},
             room=message['room'])

    def on_disconnect_request(self):
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response',
             {'data': 'Disconnected!', 'count': session['receive_count']})
        disconnect()

    def on_my_ping(self):
        emit('my_pong')

    def on_connect(self):
        global thread1
        global thread2
        global thread3
        global thread4
        global thread5
        global thread6
        with mutex:
            if thread1 is None:
                thread1 = socketio.start_background_task(yolo_process_thread)
            if thread2 is None:
                thread2 = socketio.start_background_task(ssdc_process_emit_thread)
            if thread3 is None:
                thread3 = socketio.start_background_task(getImg_thread)
            if thread4 is None:
                thread4 = socketio.start_background_task(yolo_emit_thread)
            if thread5 is None:
                thread5 = socketio.start_background_task(bluetooth_thread)
            if thread6 is None:
                thread6 = socketio.start_background_task(loc_emit_thread)
        emit('my_response', {'data': 'Connected', 'count': 0})

    def on_disconnect(self):
        print('Client disconnected', request.sid)


socketio.on_namespace(MyNamespace('/test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='0', help='source')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--cuda', type=bool,
                        default=False)
    parser.add_argument('--iou', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--debug', type=bool,
                        default=False)
    opt = parser.parse_args()
    if opt.source == '0':
        optSource = 0
    else:
        optSource = opt.source
    if opt.debug:
        bluetooth_ready = True
        
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
