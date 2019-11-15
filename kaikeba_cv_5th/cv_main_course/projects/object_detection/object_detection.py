import cv2
import numpy as np


def main():
    # import the object image and source image
    obj = cv2.imread('obj.jpg')
    src_img = cv2.imread('src_img.jpg')
    sift = cv2.xfeatures2d.SIFT_create()
    # find key points in input images
    kp_obj, des_obj = sift.detectAndCompute(obj, None)
    kp_src, des_src = sift.detectAndCompute(src_img, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # generate 5 trees, for search parallelly to make searching process fast
    search_params = dict(checks=50) # set recursion time
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # create flann based matcher
    matches = flann.knnMatch(des_obj, des_src, k=2)  # k-NearestNeighbor algorithm to search match point pairs
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # compute homography matrix, perspective transform, draw points pairs
    MIN_MATCH_COUNT = 4  # perspective transform homography matrix need at lest 4 pairs of points to compute
    if len(good) > MIN_MATCH_COUNT:
        obj_pts = np.float32([kp_obj[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # reshape points to an n*1*2 array
        src_pts = np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(obj_pts, src_pts, cv2.RANSAC, 5.0)  # use RANSAC to remove wrong matches, 5.0 as the threshold for outlier
        matchesMask = mask.ravel().tolist()  # flattern the mask matrix and transform it to a list
        h, w, c = obj.shape  # get height, width, channel of the object
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)  # four apexes of object
        dst = cv2.perspectiveTransform(pts, M)  # apexes points perspective transform with the homography matrix
        src_img = cv2.polylines(src_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  # draw the transformed outline of object on source image
        # draw match pairs of two input images on result image
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                          flags=2)
        result = cv2.drawMatches(obj, kp_obj, src_img, kp_src, good, None, **draw_params)
        cv2.imshow('object detected', result)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()