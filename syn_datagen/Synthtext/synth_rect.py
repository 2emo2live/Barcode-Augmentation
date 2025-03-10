import cv2
import numpy as np
from random import randint


def create_rect():
    # Создаем черное изображение
    width, height = 400, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Задаем цвет прямоугольника (BGR)
    b = randint(0, 255)
    g = randint(0, 255)
    r = randint(0, 255)
    color = (b, g, r)

    # Задаем координаты и размеры прямоугольника
    x1, y1 = 0, 0
    x2, y2 = 400, 300

    # Рисуем прямоугольник
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    return image


def start():
    iter = 1000
    f = open('C:/Users/phone/2D_bar_codes/syn_datagen/SRNet-Datagen/Synthtext/data/data_new/bg_path.txt', 'a')
    for i in range(80, iter):
        img = create_rect()
        cv2.imwrite(f'./data/data_new/bg_img/rect/rect_{i}.png', img)

        f.write(f"C:/Users/phone/2D_bar_codes/syn_datagen/SRNet-Datagen/Synthtext/data/data_new/bg_img/rect/rect_{i}.png\n")
