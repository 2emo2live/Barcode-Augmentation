from skimage.util import random_noise
import numpy as np
import cv2


def add_poisson_noise(image, alpha=0.05):
    """
    Добавляет шум Пуассона к изображению.
    :param image: Входное изображение.
    :return: Изображение с добавленным шумом.
    """
    noisy = random_noise(image, mode='poisson')
    noisy = (255 * noisy).astype(np.uint8)  # Преобразуем обратно в uint8
    return cv2.addWeighted(image, 1 - alpha, noisy, alpha, 0)
