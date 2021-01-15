import cv2
import os
import glob
import json
import shutil
import logging
import numpy as np
import random
import pygame

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def p_d(arr, thortl, bhorbr, decision, mode):
    """
    进行padding或者delete
    :param arr:数组
    :param thortl:高度上或者宽度左
    :param bhorbr:高度下或者宽度右
    :param decision:方向 高度或者 宽度
    :param mode:padding还是deleting
    :return:
    """
    if len(arr.shape) == 3:
        arr = cv2.split(arr)
    else:
        arr = [arr]

    if mode == "p":
        if decision == "h":
            size = ((thortl, bhorbr), (0, 0))
        else:
            size = ((0, 0), (thortl, bhorbr))

        if len(arr) == 1:
            return np.pad(arr[0], size, 'constant', constant_values=0)
        else:
            for i in range(len(arr)):
                arr[i] = np.pad(arr[i], size, 'constant', constant_values=0)
            return cv2.merge(arr)
    else:
        if decision == "h":
            ax = 0
        else:
            ax = 1

        if len(arr) == 1:
            for start in [thortl, bhorbr]:
                if start == thortl:
                    arr[0] = np.delete(arr[0], [0], axis=ax)
                else:
                    arr[0] = np.delete(arr[0], [-1], axis=ax)
                return arr[0]
        else:
            for i in range(len(arr)):
                for _ in range(thortl):
                    arr[i] = np.delete(arr[i], [0], axis=ax)
                for _ in range(bhorbr):
                    arr[i] = np.delete(arr[i], [-1], axis=ax)
            # return create_alpha(arr)
            return cv2.merge(arr)


def create_alpha(arr):
    b, g, r = cv2.split(arr)
    a = np.ones(b.shape, dtype=b.dtype) * 255
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if (b[i][j] == 0 and g[i][j] == 0 and r[i][j] == 0):
                a[i][j] = 0
    return cv2.merge((b, g, r, a))


def is_num_ok(num):
    if num % 2 == 0:
        thortl = bhorbr = int(num / 2)
        return thortl, bhorbr
    else:
        thortl = int(num / 2)
        bhorbr = thortl + 1
        return thortl, bhorbr


def resize_label(num_1, num_2):
    if num_1 > num_2:
        num = num_1 - num_2
        mode = 1  # del
        thortl, bhorbr = is_num_ok(num)
        return thortl, bhorbr, mode
    elif num_1 < num_2:
        num = num_2 - num_1
        mode = 2  # pad
        thortl, bhorbr = is_num_ok(num)
        return thortl, bhorbr, mode
    else:
        return 0, 0, 0


def show_pic(winname, mat):
    cv2.imshow(winname, mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    label_gt_path = "s2_label_gt"
    bg_gt_path = "s3_bg_gt"
    template_path = "s4_templates"
    saved_path = "s4_gd_datas"
    cache_folder = ".cache"
    json_path = glob.glob(os.path.join(template_path, "*.json"))

    for p in [saved_path, cache_folder]:
        if not os.path.exists(p):
            os.makedirs(p)
        else:
            shutil.rmtree(p)
            os.makedirs(p)

    label_gt = glob.glob(os.path.join(label_gt_path, "*.png"))
    bg_gt = glob.glob(os.path.join(bg_gt_path, "*.bmp"))

    anno = json.load(open(json_path[0], "rb"))
    anno = list(anno.values())

    for an in anno:
        image = cv2.imread(os.path.join(template_path, an['filename']))
        bg_gt_pic = cv2.imread(random.choice(bg_gt))
        # bg_gt_pic = pygame.image.load(random.choice(bg_gt))
        bg_gt_pic = cv2.resize(bg_gt_pic, (image.shape[1], image.shape[0]))
        bg_cache = os.path.join(cache_folder, "bg_gt_pic.bmp")
        cv2.imwrite(bg_cache, bg_gt_pic)
        bg_gt_pic = pygame.image.load(bg_cache)
        for reg in an['regions']:
            rect = reg['shape_attributes']
            x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']

            label_rect = pygame.Rect(0, 0, w, h)
            # resize为标注大小
            label_gt_pic = cv2.imread(random.choice(label_gt))

            l_h, l_w = label_gt_pic.shape[:2]

            for index, (a, b) in enumerate(zip([l_h, l_w], [h, w])):
                thortl, bhorbr, mode = resize_label(a, b)
                if index == 0:
                    decision = "h"
                else:
                    decision = "w"
                if mode == 1:
                    label_gt_pic = p_d(label_gt_pic, thortl, bhorbr, decision, "d")
                elif mode == 2:
                    label_gt_pic = p_d(label_gt_pic, thortl, bhorbr, decision, "p")
                else:
                    pass

            label_cache = os.path.join(cache_folder, "label_gt_pic.png")
            # label_gt_pic = cv2.blur(label_gt_pic, (3, 3))
            # cv2.imshow("label_gt_pic", label_gt_pic)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            label_gt_pic = create_alpha(label_gt_pic)
            cv2.imwrite(label_cache, label_gt_pic)
            label_gt_pic = pygame.image.load(label_cache)
            label_gt_pic.set_colorkey((0, 0, 0))
            bg_gt_pic.blit(label_gt_pic, (x, y), label_rect)

        pygame.image.save(bg_gt_pic, os.path.join(saved_path, "test.png"))

    shutil.rmtree(cache_folder)
