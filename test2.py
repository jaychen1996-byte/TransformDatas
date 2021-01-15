import numpy as np
import cv2


def show_pic(winname, mat):
    cv2.imshow(winname, mat)
    cv2.waitKey()
    cv2.destroyAllWindows()


def p_d(arr, thortl, bhorbr, decision, mode):
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
            return create_alpha(arr)
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
            return create_alpha(arr)


def create_alpha(arr):
    b, g, r = arr[:]
    a = np.ones(b.shape, dtype=b.dtype) * 255
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if (b[i][j] == 0 and g[i][j] == 0 and r[i][j] == 0):
                a[i][j] = 0
    return cv2.merge((b, g, r, a))


if __name__ == '__main__':
    image = cv2.imread("s2_label_gt/0.png")
    print(image.shape)
    show_pic("image", image)
    image_p = p_d(image, 2, 2, "h", "p")
    print(image_p.shape)
    show_pic("image_p", image_p)
    cv2.imwrite("test_pad.png", image_p)
    image_d = p_d(image, 2, 2, "h", "d")
    print(image_d.shape)
    show_pic("image_d", image_d)
    cv2.imwrite("test_del.png", image_d)
