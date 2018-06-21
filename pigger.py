from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans

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
nindex  = 0

count = 1
while count < index:
    video.read()
    count += 1


succ, img = video.read()
kmeans = KMeans(n_clusters = 1)

readouts = []

digitHull = None
digitHullSkewed = None

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
            circleWidth, circleHeight = cv2.boundingRect(contour)[2:]
            if circleWidth/circleHeight > 0.6 and circleWidth/circleHeight < 1.4:
                #print()
                #print(circleWidth, circleHeight)
                break
    else:
        print("  %d - dot not found" % index)
        readouts.append("null")
        index += 1
        continue

    (x, y), radius = cv2.minEnclosingCircle(contour)
    height, width = img.shape[:2]
    numbers = img[0:height, int(x+radius*2):width]

    numbers = cv2.copyMakeBorder(numbers, 0, 0, 3, 0, cv2.BORDER_CONSTANT, (255, 255, 255))

    height, width = numbers.shape[:2]
    numberZer = img[0:height, int(x-width/3):int(x)]
    numberOne = numbers[0:height, 0:int(width/3)]
    numberTwo = numbers[0:height, int(width/3):int(2*width/3)]
    numberTre = numbers[0:height, int(2*width/3):width]
    cx = x

    drawn = img.copy()
    cv2.drawContours(drawn, [contour], -1, (0, 255, 0), 1)
    cbx, cby, cbw, cbh = cv2.boundingRect(contour)
    cv2.rectangle(drawn, (cbx, cby), (cbx+cbw, cby+cbh), (0, 0, 255), 1)
    cv2.imwrite("output/pic%d.png" % index, drawn)

    digits = []
    for ind, number in enumerate([numberZer, numberOne, numberTwo, numberTre]):
        greyber = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
        greyber = cv2.threshold(greyber, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        try:
            assert(greyber.shape[1]>0 and greyber.shape[0]>0)
        except:
            digits = []
            break

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 6))
        distinguished = cv2.morphologyEx(greyber, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        greyber = cv2.erode(greyber, kernel, iterations = 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 6))
        greyber = cv2.dilate(greyber, kernel, iterations = 3)
        greyber = cv2.erode(greyber, kernel, iterations = 1)

        if digitHull == None:
            #this relies on the assumption that the first digit processed will be a 0
            ret, contours, hierarchy = cv2.findContours(greyber.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
            borderHull = cv2.convexHull(contours[0])
            digitHull = borderHull

            ret, contours, hierarchy = cv2.findContours(greyber.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
            borderHull = cv2.convexHull(contours[0])
            digitHullSkewed = borderHull

        hullBounds = cv2.boundingRect(digitHullSkewed)

        greyber = distinguished[
                hullBounds[1]:hullBounds[1]+hullBounds[3],
                hullBounds[0]:hullBounds[0]+hullBounds[2]]
        try:
            greyber = cv2.dilate(greyber, kernel, iterations = 1)
        except:
            cv2.imshow("num" + str(index), number)
            cv2.waitKey()
            print(greyber.shape)
            raise SystemExit
        borderAmt = (hullBounds[3]-hullBounds[2])//2
        greyber = cv2.copyMakeBorder(greyber, 0, 0, borderAmt, borderAmt, cv2.BORDER_CONSTANT, (0, 0, 0))

        height, width = greyber.shape[:2]


        pts1 = np.float32([[5, 5], [width-5, 5], [5, height-5]])
        pts2 = np.float32([[2, 5], [width-5-3, 5], [5, height-5]])

        afftr = cv2.getAffineTransform(pts1, pts2)
        greyber = cv2.warpAffine(greyber, afftr, greyber.shape[:2])
        greyber = greyber[0:height, 
                borderAmt:width-borderAmt
        ]
        #cv2.imshow("greyber", greyber)


        # margins (left, right, top, bottom)
        lm = 0
        rm = 2
        tm = 2
        bm = 4
        (x, y, w, h) = (lm, tm, greyber.shape[1]-lm-rm, greyber.shape[0]-tm-bm)
        dw = int(w*0.35)
        dh = int(h*0.15)

        # define the set of 7 segments
        segments = [
                ((x, y+dh), (x+dw, y+h // 2)), # top-left
                ((x, y+h // 2), (x+dw, y+h-dh)), # bottom-left
                ((x+dw, y), (x+w-dw, y+dh)),      # top
                ((x+dw, (y+(h-dh) // 2)) , (x+w-dw, y+((h+dh) // 2))), # center
                ((x+dw, y+h - dh), (x+w-dw, y+h)),   # bottom
                ((x+w-dw, y+dh), (x+w, y+h // 2)),     # top-right
                ((x+w-dw, y+h // 2), (x+w, y+h-dh))     # bottom-right
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
            #print(total/area)
            if total / area > -0.001607*area + 0.5:
                on[i] = 1
                color = (0, 255, 0)
            else:
                pass
            cv2.rectangle(copy, segment[0], segment[1], color, 1)
            if index > 9990:
                print()
                print(total, area, total/area)
                cv2.imshow("sector", copy)
                cv2.waitKey()

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

    if value == "null":
        cv2.imshow(str(index) + " - null", img)
        cv2.moveWindow(str(index) + " - null", (nindex * 314) % 1884, int(nindex * 314 / 1884) * 105)
        nindex += 1

    #print(" ", index, "-", int(value*1000) if len(digits) > 0 else "null")
    readouts.append(value)

    index += 1

    succ, img = video.read()

else:
    print("done                  ")
index -= 1
print(index, len(readouts))

wasNull = False

outputFile = open("output/" + filename.split("/")[-1] + ".out", "w")
outString = ""
for i, val in enumerate(readouts):
    if val == "null":
        wasNull = True
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

if wasNull:
    cv2.waitKey()
