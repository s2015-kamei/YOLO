import os
import cv2

img_dir = '/home/roboken03/Downloads/Data/' # labelImgで結果を出力したディレクトリ

def main():

    for file in os.listdir(img_dir):
        name, ext = os.path.splitext(file)
        print('base:' + name + '  ext:' + ext)
        if ext == '.png':
            img = cv2.imread(img_dir + file)
            label_info = load_labeldata(name)
            flipped_y(img, name, label_info)
            flipped_x(img, name, label_info)
            flipped_xy(img, name, label_info)

def flipped_y(img, filenm_base, label_info):
    img_flip_ud = cv2.flip(img, 0)
    cv2.imwrite(img_dir + filenm_base + '_flipped_y.png', img_flip_ud)

    f = open(img_dir + filenm_base + '_flipped_y.txt', 'a')
    for data in label_info:
        label, x_coordinate, y_coordinate, x_size, y_size = data.split()
        f.write(label + ' ' + x_coordinate + ' ' + turn_over(y_coordinate) + ' ' + x_size + ' ' + y_size + '\n')
    f.close()

def flipped_x(img, filenm_base, label_info):
    img_flip_lr = cv2.flip(img, 1)
    cv2.imwrite(img_dir + filenm_base + '_flipped_x.png', img_flip_lr)

    f = open(img_dir + filenm_base + '_flipped_x.txt', 'a')
    for data in label_info:
        label, x_coordinate, y_coordinate, x_size, y_size = data.split()
        f.write(label + ' ' + turn_over(x_coordinate) + ' ' + y_coordinate + ' ' + x_size + ' ' + y_size + '\n')
    f.close()

def flipped_xy(img, filenm_base, label_info):
    img_flip_ud_lr = cv2.flip(img, -1)
    cv2.imwrite(img_dir + filenm_base + '_flipped_xy.png', img_flip_ud_lr)

    f = open(img_dir + filenm_base + '_flipped_xy.txt', 'a')
    for data in label_info:
        label, x_coordinate, y_coordinate, x_size, y_size = data.split()
        f.write(label + ' ' + turn_over(x_coordinate) + ' ' + turn_over(y_coordinate) + ' ' + x_size + ' ' + y_size + '\n')
    f.close()

def turn_over(coordinate):
    val = 1 - float(coordinate)
    return '{:.6f}'.format(val)


def load_labeldata(filenm):
    try:
        f = open(img_dir + filenm + '.txt', 'r')
        return f.readlines()
    except Exception as e:
        print(filenm)
        print(e)
    finally:
        f.close()

main()