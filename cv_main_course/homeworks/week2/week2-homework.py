# 1.Code a function to do median blur operations by your self.
import cv2
import numpy as np
import time


def median_blur(img, ker_size):
    height, width, ch_num = img.shape
    channels = [[]] * ch_num
    for i in range(ch_num):
        channels[i] = cv2.split(img)[i]
    for channel in channels:
        for i in range(height):
            for j in range(width):
                channel[i][j] = comp_median(get_ker_array(channel, height, width, i, j, ker_size))
    return cv2.merge(channels)


def get_ker_array(mat, height, width, xpos, ypos, ksize):
    return [mat[x][y] for x in range(xpos-ksize//2, xpos+ksize//2+1) if 0 <= x < width
                       for y in range(ypos-ksize//2, ypos+ksize//2+1) if 0 <= y < height]


def comp_median(src_list):
    #return np.median(src_list)
    return merge_sort(src_list)[len(src_list)//2]

    
def merge_sort(in_list):
    n = len(in_list)
    if n <= 1:
        return in_list
    else:
        l1, l2 = in_list[:n//2], in_list[n//2:]
        return merge(merge_sort(l1), merge_sort(l2))


def merge(l1, l2):
    n1, n2, i1, i2 = len(l1), len(l2), 0, 0
    n = n1 + n2
    l = [0] * n
    for i in range(n):
        if i2 > n2-1 or (i1 <= n1-1 and l1[i1] < l2[i2]):
            l[i] = l1[i1]
            i1 += 1
        else:
            l[i] = l2[i2]
            i2 += 1
    return l


def main():
    input_img = cv2.imread('D:\\lenna.jpg', 1)
    start1 = time.perf_counter()
    output_img = median_blur(input_img, 7)
    elapsed1 = (time.perf_counter() - start1)
    start2 = time.perf_counter()
    md_img = cv2.medianBlur(input_img, 7)
    elapsed2 = (time.perf_counter() - start2)
    print("My median blur time cost:%ss \nOpenCV median blur time cost:%ss" % (elapsed1,elapsed2))
    cv2.imshow('lenna',input_img)
    cv2.imshow('lenna_md_my',output_img)
    cv2.imshow('lenna_md_opencv',md_img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()



# 2.Read RANSAC algorithm and write a pseudo code about it.
'''
Input:
dataset: the set of data to be explained by a model 
model: a model to fit the input dataset, need at lest n parameters
threshold: the threshold to determine if the data can fit in the model 
max_iter: max number of iterations
d: 
Output:
best_data: best data from the input dataset as the parameters to compute the model

Begin
iter := 0
best_data := None
best_inliers_numer := 0
while iter < max_iter {
    inliers := randomly choose n data values from dataset
    new_model := model compute by inliers
    for data in dataset and not in inliers {
        if data error for new_model < threshold {
            inliers := inliers add data
            }
        }
    if the number of data in inliers > best_inliers_number {
        best_model := new_model
        best_inliers_numer := the number of data in inliers
        } 
    iter := iter + 1
    }
best_data := dataset fit the best_model
return best_data
End
'''
