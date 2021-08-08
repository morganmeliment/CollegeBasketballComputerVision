import numpy as np
import cv2
from matplotlib import pyplot as plt
"""
img = cv2.imread('f2.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img2 =cv2.drawKeypoints(gray,kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoints.jpg',img2)

"""

"""
filename = 'gameShot copy.png'
img = cv2.imread(filename)
gray = cv2.Canny(cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2HSV))[1], 350, 430, apertureSize = 3)


gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
masking = (dst > 0.01*dst.max()) #& (dst < 0.03*dst.max())
gray[masking]=100

cv2.imshow('dst',gray)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
"""


img = cv2.imread('f2.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,5,0.01,5)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()

"""
img = cv2.imread('gameShot.png',0)
equ = cv2.equalizeHist(img)
equ[equ > 30] = 255
img[equ < 31] = 0
img1 = cv2.Canny(equ, 350, 430, apertureSize = 3)
plt.imshow(img),plt.show()
"""
"""
MIN_MATCH_COUNT = 1

img1 = cv2.imread('b1g.png',0)          # queryImage
img2 = cv2.imread('gameShot copy 4.png', 0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 500)

flann = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = flann.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

# store all the good matches as per Lowe's ratio test.
N_MATCHES = 10

match_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:N_MATCHES], img2.copy(), flags=0)

plt.figure(figsize=(12,6))
plt.imshow(match_img)
plt.show()
"""





