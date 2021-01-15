import cv2
import os
import glob
import json
import shutil
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def show_pic(winname, mat):
    cv2.imshow(winname, mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    saved_path = "s2_label_gt"
    files_path = "s1_rotated_output"
    json_path = glob.glob(os.path.join(files_path, "*.json"))

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    else:
        shutil.rmtree(saved_path)
        os.makedirs(saved_path)

    anno = json.load(open(json_path[0], "rb"))
    anno = list(anno.values())

    datas = open(os.path.join(saved_path, "dict.txt"), "w")
    count = 0
    for an in anno:
        image = cv2.imread(os.path.join(files_path, an['filename']), 0)
        for reg in an['regions']:
            # print(reg['shape_attributes'])
            rect = reg['shape_attributes']
            x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
            section = image[y:y + h, x:x + w]

            ret, section_th = cv2.threshold(section, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            section = cv2.bitwise_and(section, section_th, mask=section_th)

            # 创建四通道图片
            b, g, r = cv2.split(cv2.cvtColor(section, cv2.COLOR_GRAY2BGR))
            a = np.ones(b.shape, dtype=b.dtype) * 255
            for i in range(section.shape[0]):
                for j in range(section.shape[1]):
                    if (b[i][j] == 0 and g[i][j] == 0 and r[i][j] == 0):
                        a[i][j] = 0
            img_al = cv2.merge((b, g, r, a))

            label = reg['region_attributes']['name']
            cv2.imwrite(os.path.join(saved_path, label + ".png"), img_al)
            datas.write(label + "\n")
            count += 1
    logging.info("完成标签切割,共有%d个标签！" % count)
    datas.close()
