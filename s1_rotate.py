from PIL import Image
import os
import shutil
import logging

logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                    level=logging.INFO)


def rotate(in_path, out_path, file_name):
    angle = 0
    while True:
        line = input()

        num = [str(i) for i in range(181)]
        num += [str(-i) for i in range(181)]
        num_total = num + ["a", "q", "d", "s", "r"]

        if line not in num_total:
            logging.info("输入单字符操作！'q'退出,'a'顺时针旋转,'d'逆时针旋转.")

        if line == "q":
            break

        img = Image.open(in_path)

        if line == "a":
            angle -= 1
            img_rotate = img.rotate(angle, resample=Image.BICUBIC, expand=0)
            logging.info("顺时针旋转%d度！" % abs(angle))

        if line == "d":
            angle += 1
            img_rotate = img.rotate(angle, resample=Image.BICUBIC, expand=0)
            logging.info("逆时针旋转%d度！" % abs(angle))

        if line == "r":
            angle = 0
            img_rotate = img
            logging.info("重置图像！")

        if line == "s":
            img_rotate.save(os.path.join(out_path, file_name))
            logging.info("保存图像！当前旋转角度为%d度!" % angle)
            break

        if line in num:
            angle = int(line)
            img_rotate = img.rotate(int(line), resample=Image.BICUBIC, expand=0)

        img_rotate.show()


if __name__ == '__main__':
    images_path = "s1_label_gt_images"
    output_path = "s1_rotated_output"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    for index, image in enumerate(os.listdir(images_path)):
        logging.info("处理第%d张图像！" % int(index + 1))
        rotate(os.path.join(images_path, image), output_path, image)
