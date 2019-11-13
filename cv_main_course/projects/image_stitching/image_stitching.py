import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt


def main():
    # import two images
    img1 = cv2.imread('test1.jpg')
    img2 = cv2.imread('test2.jpg')
    # show the input images
    cv2.imshow('first image', img1)
    cv2.imshow('second image', img2)
    ## a simple way
    ## stitcher = cv2.createStitcher(False)
    ## _result, pano = stitcher.stitch((img1, img2))
    ## cv2.imshow('pano', pano)
    # create SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find key points in input images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # draw key points on input images
    img1_kp = cv2.drawKeypoints(img1, kp1, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2, kp2, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('first image with key points', img1_kp)
    cv2.imshow('second image with key points', img2_kp)
    # use k-d tree algorithm to match key points
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # generate 5 trees, for search parallelly to make searching process fast
    search_params = dict(checks=50) # set recursion time
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # create flann based matcher
    matches = flann.knnMatch(des1, des2, k=2)  # k-NearestNeighbor algorithm to search match point pairs
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # compute homography matrix, perspective transform, draw points pairs
    MIN_MATCH_COUNT = 4  # perspective transform homography matrix need at lest 4 pairs of points to compute
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # reshape points to an n*1*2 array
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # use RANSAC to remove wrong matches, 5.0 as the threshold for outlier
        matchesMask = mask.ravel().tolist()  # flattern the mask matrix and transform it to a list
        # draw match pairs of two input images on image3
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        cv2.imshow('matched images', img3)
        # shift image2 to right for the width of the image1
        shift = np.array([[1.0, 0, img1.shape[1]], [0, 1.0, 0], [0, 0, 1.0]])  # make shift matrix
        img2 = cv2.warpPerspective(img2, shift, (img1.shape[1] * 2, img2.shape[0]))  # shift image by affine transform
        #cv2.imshow('shifted image2', img2)
        # transform images1 and blend with image2 to generate image4
        M = np.dot(shift, M)  # shift matrix dot multiply by homography matrix
        img1 = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))  # perspective transform image1
        #cv2.imshow('trans image', img1)
        result = img2  # generate result image which have the same size of image2
        result[:,:img1.shape[1]//2] = img1[:,:img1.shape[1]//2]  # left side of the result image is image1
        cv2.imshow('result image', result)  # show result image
        cv2.imwrite('result.jpg' , result)  # save result image
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()
