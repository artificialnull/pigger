from imutils.perspective import four_point_transform
from imutils import contours as contourlib
from math import sqrt
import imutils
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", required=True, help="video path")
args = vars(ap.parse_args())
filename = args["f"]
print(filename)

video   = cv2.VideoCapture(filename)
data    = open("output/" + filename.split("/")[-1] + ".out").read()
data    = data.split('\n')

nums = []
for i, line in enumerate(data):
    if len(line) > 0 and line != "ERROR":
        nums.append(float(line))
    else:
        print("discrepency", i)

index   = 1

count = 1
while count < index:
    video.read()
    count += 1

succ, img = video.read()

last = 0

frames = {}

while succ:

    if nums[index - 1] != last:
        img = img[0:540, 200:1200]

        cv2.imshow("measurement", img)
        digits = []
        while True:
            imc = img.copy()
            val = cv2.waitKey()
            if val == 8:
                digits = digits[:-1]
            elif val == 13 or val == ord(" "):
                break
            elif val == ord("f"):
                while succ:
                    succ, img = video.read()
                break
            else:
                digits.append(chr(val))
            cv2.putText(imc, "".join(digits), (200, 300), 0, 1, (0, 255, 0))
            cv2.imshow("measurement", imc)

        if len(digits) == 0:
            continue

        num = float("".join(digits))
        print("FINAL:", num)
        frames[index] = (num, nums[index - 1])
        print(num, nums[index - 1])
        last = nums[index - 1]

    succ, img = video.read()
    index += 1

else:
    print("done")

output = open("output/" +  filename.split("/")[-1] + ".final", "w")
outstr = ""
for i in range(0, max(frames.keys()) + 1):
    if i in frames.keys():
        outstr += str(frames[i][0]) + ", " + str(frames[i][1]) + "\n"
output.write(outstr)
output.close()
