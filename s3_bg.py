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
    saved_path = "s3_bg_gt"
    files_path = "s3_bg_images"
    json_path = glob.glob(os.path.join(files_path, "*.json"))

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    else:
        shutil.rmtree(saved_path)
        os.makedirs(saved_path)

    anno = json.load(open(json_path[0], "rb"))
    anno = list(anno.values())
    count = 0
    for an in anno:
        image = cv2.imread(os.path.join(files_path, an['filename']), 0)
        for reg in an['regions']:
            rect = reg['shape_attributes']
            x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']

            fake_image = np.full(image.shape, 255, dtype=np.uint8)
            section = image[y:y + h, x:x + w]

            ret, section_th = cv2.threshold(section, 0, 255, cv2.THRESH_OTSU)
            fake_image[y:y + h, x:x + w] = section_th
            fake_image_af = cv2.bitwise_and(image, fake_image, mask=fake_image)
            inpaint_mask = 255 - fake_image
            kernel = np.ones((3, 3), np.uint8)
            inpaint_mask_dilate = cv2.dilate(inpaint_mask, kernel)
            # dst_TELEA = cv2.inpaint(fake_image_af, inpaint_mask_dilate, 3, cv2.INPAINT_TELEA)
            dst_NS = cv2.inpaint(fake_image_af, inpaint_mask_dilate, 3, cv2.INPAINT_NS)
            cv2.imwrite(os.path.join(saved_path, an['filename']), dst_NS)
            count += 1
    logging.info("完成背景修复,共有%d个背景图！" % count)
