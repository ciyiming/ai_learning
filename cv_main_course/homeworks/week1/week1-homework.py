# Combine image crop, color shift, rotation and
# perspective transform together to complete a
# data augmentation script
import cv2
import random
import numpy as np


def image_crop(img):
    '''
    randomly crop the input image
    :param img: the input image
    :return: randomly croped image
    '''
    height, width, _ = img.shape    # get the height and width of the input image
    cols_start = random.randint(0, width//3)
    cols_end = random.randint(cols_start+width//3, width-1)
    rows_start = random.randint(0, height//3)
    rows_end = random.randint(rows_start+height//3, height-1)
    img_crop = img[rows_start:rows_end, cols_start:cols_end]    # crop image through matrices slice
    return img_crop


def image_color_shift(img):
    '''
    randomly shift the color of the input image
    :param img: the input image
    :return: color shifted image
    '''
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    # convert the color space from BGR to YUV
    Y, U, V = cv2.split(img_yuv)    # split the image to 3 channels
    u_shifter = random.randint(-224, 224)    # U channel, V channel vary from 16 to 240, the shift range is -224~224
    v_shifter = random.randint(-224, 224)
    if u_shifter >= 0:
        lim = 240 - u_shifter
        U[U > lim] = 240
        U[U <= lim] = (u_shifter + U[U <= lim]).astype(img.dtype)
    elif u_shifter < 0:
        lim = 16 - u_shifter
        U[U < lim] = 16
        U[U >= lim] = (u_shifter + U[U >= lim]).astype(img.dtype)
    if v_shifter >= 0:
        lim = 240 - v_shifter
        V[V > lim] = 240
        V[V <= lim] = (v_shifter + V[V <= lim]).astype(img.dtype)
    elif v_shifter < 0:
        lim = 16 - v_shifter
        V[V < lim] = 16
        V[V >= lim] = (v_shifter + V[V >= lim]).astype(img.dtype)
    img_tmp = cv2.merge((Y, U, V))
    img_shift = cv2.cvtColor(img_tmp, cv2.COLOR_YUV2BGR)
    return img_shift


def image_rotation(img):
    '''
    randomly rotate the input image
    :param img: the input image
    :return: rotated image
    '''
    angle = random.randint(-180, 180)    # randomly choose an angle
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)    # compute the transform matrix
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))    # rotate the input image
    return img_rotate


def perspective_transform(img):
    '''
    randomly transform perspective of the input image
    :param img: the input image
    :return: transformed image
    '''
    height, width, _ = img.shape
    xs = [random.randint(0, width - 1) for _ in range(4)]  # randomly choose 4 points (x,y) from input image
    ys = [random.randint(0, height - 1) for _ in range(4)]
    dxs, dys = [], []
    for x in xs:
        dxs.append(x+random.randint(-2,2))   # choose the destination points within 2 pixels around the source points
    for y in ys:
        dys.append(y+random.randint(-2,2))
    pts1 = np.float32(list(zip(xs, ys)))
    pts2 = np.float32(list(zip(dxs, dys)))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_trans = cv2.warpPerspective(img, M, (width, height))
    return img_trans


def main():
    src_img_name = input("Please input source image file direction and file name:")
    count = input("Please input the number of images to generate:")
    dst_dir = input("Please input the output direction:")
    input_img = cv2.imread(src_img_name, 1)
    for i in range(int(count)):
        pro_type = random.randint(0, 3)
        output_img = input_img
        if pro_type == 0:
            output_img = image_crop(input_img)
        elif pro_type == 1:
            output_img = image_color_shift(input_img)
        elif pro_type == 2:
            output_img = image_rotation(input_img)
        elif pro_type == 3:
            output_img = perspective_transform(input_img)
        cv2.imwrite('%s\\%s.jpg'%(dst_dir,i), output_img)
    return


if __name__ == "__main__":
    main()
