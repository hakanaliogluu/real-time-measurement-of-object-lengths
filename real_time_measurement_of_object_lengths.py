from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

mussles = []

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

kamera = cv2.VideoCapture(0)

while True:
    ret, file_path = kamera.read()

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--file_path", required=False,
                    help="path to the input file_path", default= file_path)
    ap.add_argument("-w", "--width", type=float, required=False,
                    help="width of the left-most object in the file_path (in inches)", default= '0.9')
    args = vars(ap.parse_args())

    gray = cv2.cvtColor(file_path, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None


    print("pre filter count: ", len(cnts))

    #cnts = [c for c in cnts if cv2.contourArea(c) > 90 ]
    print("Post-filter count: ", len(cnts))

    for idx, c in enumerate(cnts):

        orig = file_path.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        for (x, y) in box:

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)


            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


            if pixelsPerMetric is None:
                    pixelsPerMetric = (dB / args["width"])

            dimA = (dA / pixelsPerMetric) * 2.54
            dimB = (dB / pixelsPerMetric) * 2.54

            if dimA >=1 and dimB >=1:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                cv2.putText(orig, "{:.1f}cm".format(dimB),
                            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (100, 100, 100), 2)
                cv2.putText(orig, "{:.1f}cm".format(dimA),
                            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (100, 100, 100), 2)
                mussle = {'count' : idx, 'dimA': dimA, 'dimB': dimB}
                mussles.append(mussle)
                print(mussle)
 
                cv2.imshow("Image", orig)
                cv2.waitKey(0)

            else:
                continue
            
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
            
kamera.release()

cv2.destroyAllWindows()
