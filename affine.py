import cv2
import numpy as np

img = cv2.imread('rec.png')
rows,cols,ch = img.shape

pts1 = np.float32([[10,0],[300,0],[0,300]])
pts2 = np.float32([[0,0],[290,0],[0,300]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow("rect", dst)
cv2.waitKey()
