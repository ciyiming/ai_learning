import cv2
import numpy as np


def draw_box(event, x, y, flag, param):
    global x1, y1, x2, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        cv2.rectangle(first_frame, (x1, y1), (x, y), (0, 0, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(first_frame, (x1, y1), (x, y), (0, 0, 255), 2)
        x2, y2 = x, y


def object_detection(obj, src_img):
    orb = cv2.ORB_create()
    kp_obj, des_obj = orb.detectAndCompute(obj, None)
    kp_src, des_src = orb.detectAndCompute(src_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_obj, des_src)
    matches = sorted(matches, key=lambda x: x.distance)[:len(matches)//4]
    MIN_MATCH_COUNT = 4
    if len(matches) > MIN_MATCH_COUNT:
        obj_pts = np.float32([kp_obj[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp_src[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(obj_pts, src_pts, cv2.RANSAC, 1.5)
        h, w, c = obj.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(src_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))
    return


if __name__ == '__main__':
    x1, y1, x2, y2 = 0, 0, 0, 0
    msg = 'draw box, press enter to confirm'
    cv2.namedWindow(msg)
    cv2.setMouseCallback(msg, draw_box)
    cap = cv2.VideoCapture('test.mp4')
    ret, first_frame = cap.read()
    while True:
        cv2.imshow(msg, first_frame)
        if cv2.waitKey(1) & 0xFF == 13:
            break
    obj = first_frame[y1:y2, x1:x2]
    cv2.destroyAllWindows()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            object_detection(obj, frame)
            cv2.imshow("result, press esc to quit", frame)
        else:
            print('Video end. Automatically quit.')
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
