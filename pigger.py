from imutils.perspective import four_point_transform
from imutils import contours as contourlib
from math import sqrt
import imutils
import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-f", required=True, help="video path")
args = vars(ap.parse_args())
filename = args["f"]
print(filename)

DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 0, 0, 0, 1, 1): 1,
        (0, 1, 1, 1, 1, 1, 0): 2,
        (0, 0, 1, 1, 1, 1, 1): 3,
        (1, 0, 0, 1, 0, 1, 1): 4,
        (1, 0, 1, 1, 1, 0, 1): 5,
        (1, 1, 1, 1, 1, 0, 1): 6,
        (1, 0, 1, 0, 0, 1, 1): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 0, 1, 1, 1, 1, 1): 9
}

video   = cv2.VideoCapture(filename)

index   = 1

count = 1
while count < index:
    video.read()
    count += 1


succ, img = video.read()
kmeans = KMeans(n_clusters = 1)

readouts = []

while succ:

    img = imutils.resize(img, height=500)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowerBlue = np.array([105, 175, 175])
    upperBlue = np.array([115, 255, 255])

    lowerBlack = np.array([0, 0, 0])
    upperBlack = np.array([180, 255, 39])

    blackMask = cv2.inRange(hsv, lowerBlack, upperBlack)

    mask = blackMask.copy()

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    ret, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    found = False
    chosenCnt = contours[0]
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if w > cv2.boundingRect(chosenCnt)[2] and h > cv2.boundingRect(chosenCnt)[3]:
            sector = img[y:y+h, x:x+w]
            sector = sector.reshape((sector.shape[0] * sector.shape[1], 3))

            kmeans.fit(sector)
            if kmeans.cluster_centers_.astype(int)[0][0] > 130:
                chosenCnt = cnt
                found = True

    [fx, fy, fw, fh] = cv2.boundingRect(chosenCnt)
    #cv2.rectangle(img, (x, y), (x+w, y+h), (244, 91, 12), 14)
    img = img[fy:fy+fh, fx:fx+fw]
    cv2.imwrite("output/pic%d.png" % index, img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #at this point, we have the image isolated to a rectangle of the screen
    #now time to figure out the more exact borders of the screen

    blueMask = cv2.inRange(hsv, lowerBlue, upperBlue)
    mask = blueMask.copy()

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    ret, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 0:
        readouts.append("null")
        index += 1
        succ, img = video.read()
        continue


    borderContour = np.vstack(contours)
    borderHull = cv2.convexHull(borderContour)
    peri = cv2.arcLength(borderHull, True)
    approx = cv2.approxPolyDP(borderHull, 0.05 * peri, True)

    temp = img.copy()
    cv2.drawContours(temp, approx, -1, (0, 255, 0), 3)

    img = four_point_transform(img, approx.reshape(4, 2))
    img = imutils.rotate_bound(img, 90)

    height, width = img.shape[:2]
    img = img[0:int(height*0.69), 0:width] #teehee

    #now we have a flattened image of the screen only
    #next thing is to get digits from the screen

    thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    ret, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 5 and cv2.minEnclosingCircle(contour)[1] < 5:
            break
    (x, y), radius = cv2.minEnclosingCircle(contour)
    height, width = img.shape[:2]
    numbers = img[0:height, int(x+radius*2):width]

    height, width = numbers.shape[:2]
    numberZer = img[0:height, int(x-width/3):int(x)]
    numberOne = numbers[0:height, 0:int(width/3)]
    numberTwo = numbers[0:height, int(width/3):int(2*width/3)]
    numberTre = numbers[0:height, int(2*width/3):width]
    cx = x

    digits = []
    for ind, number in enumerate([numberZer, numberOne, numberTwo, numberTre]):
        greyber = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
        greyber = cv2.threshold(greyber, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        try:
            assert(greyber.shape[1]>0 and greyber.shape[0]>0)
        except:
            digits = []
            break

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        greyber = cv2.erode(greyber, kernel, iterations = 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 6))
        greyber = cv2.dilate(greyber, kernel, iterations = 3)
        greyber = cv2.erode(greyber, kernel, iterations = 1)

        ret, contours, hierarchy = cv2.findContours(greyber.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
#        copy = greyber.copy()
#        copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2BGR)
        borderHull = cv2.convexHull(contours[0])
        hullBounds = cv2.boundingRect(borderHull)
#
#        cv2.drawContours(copy, [borderHull], -1, (0, 255, 0), 2)
#
        greyber = greyber[
                hullBounds[1]:hullBounds[1]+hullBounds[3],
                hullBounds[0]+hullBounds[2]-int(hullBounds[3]/greyber.shape[0] * greyber.shape[1]):hullBounds[0]+hullBounds[2]+1]

        #(x, y, w, h) = cv2.boundingRect(contours[0])
        (x, y, w, h) = (0, 0, greyber.shape[1], greyber.shape[0])
        dw = int(w*0.3)
        dh = int(h*0.2)
        xd = 4

        # define the set of 7 segments
        segments = [
                ((x+xd//2, y+dh), (x+dw+xd, y+h // 2)), # top-left
                ((x, y+h // 2), (x+w-2*dw, y+h-dh)), # bottom-left
                ((x+dw+xd//2, y), (x+w-dw+xd//2, y+dh)),      # top
                ((x+dw, (y+(h-dh) // 2)) , (x+w-dw, y+((h+dh) // 2))), # center
                ((x+dw, y+h - dh), (x+w-dw, y+h)),   # bottom
                ((x+w-dw, y+dh), (x+w, y+h // 2)),     # top-right
                ((x+w-dw-xd//2, y+h // 2), (x+w-xd//2, y+h-dh))     # bottom-right
        ]
        on = [0] * len(segments)
        try:
            assert(greyber.shape[1]>0 and greyber.shape[0]>0)
        except:
            print(greyber.shape)
            continue
        for i, segment in enumerate(segments):
            ((left, top), (right, bottom)) = segment
            copy = greyber.copy()
            copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2BGR)
            sector = greyber.copy()[top:bottom, left:right]
            total = cv2.countNonZero(sector)
#            print(segment)
            area  = (bottom - top) * (right - left) + 1

            color = (0, 0, 255)
            if total / area > 0.5:
                on[i] = 1
                color = (0, 255, 0)
            else:
                pass
            cv2.rectangle(copy, segment[0], segment[1], color, 2)
            #cv2.imshow("sector", copy)
            #cv2.waitKey()

        try:
            digits.append(str(DIGITS_LOOKUP[tuple(on)]))
        except KeyError:
            digits = []
            break

    if len(digits) > 0:
        value = int("".join(digits)) / (10**(len(digits)-1))
        print(" ", index, "-", int(value * 1000), "    ", "\r", end="")
        if value * 1000 > 799:
            print("?")
    else:
        value = "null"
        print(" ", index, "-", "null", "    " )
    cv2.putText(img, str(value), (0, 24), 0, 1, (255, 255, 255))
    cv2.imwrite("output/screen%d.png" % index, img)

    #print(" ", index, "-", int(value*1000) if len(digits) > 0 else "null")
    readouts.append(value)

    index += 1

    succ, img = video.read()

else:
    print("done                  ")

outputFile = open("output/" + filename.split("/")[-1] + ".out", "w")
outString = ""
for i, val in enumerate(readouts):
    if val == "null":
        if i > 0 and readouts[i-1] != "null":
            prior = readouts[i-1]
        else:
            prior = None
        if i < len(readouts) - 1 and readouts[i+1] != "null":
            after = readouts[i+1]
        else:
            after = None

        if prior != None and after != None:
            readouts[i] = (prior + after) / 2
        elif prior != None:
            readouts[i] = prior
        elif after != None:
            readouts[i] = after
        else:
            readouts[i] = "ERROR"
    #print(readouts[i])
    outString += str(readouts[i]) + "\n"

outputFile.write(outString)
outputFile.close()
