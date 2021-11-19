#!/usr/bin/python3
# update time 20201010 21:59
import pygame
# import sys
import socket
import time

print('-1')
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
print('0')
host = socket.gethostname()  # 获取本地主机名
port = 9999
serversocket.bind((host, port))  # 绑定端口号
print('1')
serversocket.listen(5)  # 设置最大连接数，超过后排队
car_ctrl = False
car_ctrl_msg = "hello"
moving_directionx = 0
moving_directiony = 0

while True:
    print('2')
    clientsocket, addr = serversocket.accept()  # 建立客户端连接
    print("连接地址: %s" % str(addr))
    car_ctrl = True

    print('3')
    msg = 'BlueToothCar Server Connected' + "\r\n"
    clientsocket.send(msg.encode('utf-8'))

    car_index = input("输入所控制小车的序号: ")
    print("Car Index:", car_index + "\r\n" + "请关闭小车控制的窗口再连接下一台蓝牙小车" + "\r\n")

    pygame.init()
    print('4')
    screen = pygame.display.set_mode([640, 480])
    pygame.display.set_caption("蓝牙小车移动窗口")
    bg_color = [255, 255, 255]
    image = pygame.image.load("C:/Users/LUCK/Desktop/2020电赛省赛/BlueToothCar/pic/车辆俯视图.png")
    rect = image.get_rect()
    screen_rect = screen.get_rect()
    rect.center = screen_rect.center

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
        clientsocket.send(car_ctrl_msg.encode('utf-8'))
    clientsocket.close()
