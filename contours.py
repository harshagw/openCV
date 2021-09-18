import cv2

from stacked_image import *

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 240)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",23,255,empty)
cv2.createTrackbar("Threshold2","Parameters",20,255,empty)
# cv2.createTrackbar("Area","Parameters",5000,30000,empty)

while True:
    ret,img = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), cv2.BORDER_DEFAULT)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    canny = cv2.Canny(blur, threshold1, threshold2)
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(canny,kernel , iterations=2)
    eroded = cv2.erode(dilated,kernel,iterations=1)

    ret, thresh = cv2.threshold(gray, threshold1, threshold2, cv2.THRESH_BINARY)

    contours, hierarchies = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contourImg = np.zeros(img.shape, dtype='uint8')
    cv2.drawContours(contourImg, contours, -1, (0,225,255), 1)

    StackedImages = stackImages(0.5,([img,blur,canny],
                                           [dilated,thresh,contourImg]))

    cv2.imshow("Staked Images", StackedImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()