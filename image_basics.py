import cv2
from stacked_image import *

def translate(img,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv2.warpAffine(img,transMat,dimensions)

def rotate(img,angle,rotPoint=None,scale=1.0):
    (height,width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv2.getRotationMatrix2D(rotPoint,angle,scale)
    dimension = (width,height)

    return cv2.warpAffine(img,rotMat,dimension)

imgOriginal = cv2.imread("assets/bg1.jpg")

img = cv2.resize(imgOriginal, (0,0), fx=0.08,fy=0.08)

imgResized = cv2.resize(imgOriginal,(1000,300))  # (width, height)
imgRotate = cv2.rotate(imgOriginal, cv2.cv2.ROTATE_180)
imgCropped = img[100:200,100:400]
imgTranslate = translate(img,100,100)
imgRotateFunc = rotate(img,45,scale=1.2)
imgFlipVert = cv2.flip(img,0)
imgFlipHori = cv2.flip(img,1)

width = img.shape[1]
height = img.shape[0]

imgDraw = img.copy()
imgDraw = cv2.line(imgDraw, (0, 0), (width, height), (255, 0, 0), 10)
imgDraw = cv2.line(imgDraw, (0, height), (width, 0), (0, 255, 0), 5)
imgDraw = cv2.rectangle(imgDraw, (100, 100), (200, 200), (128, 128, 128), 5)
imgDraw = cv2.circle(imgDraw, (300, 300), 60, (0, 0, 255), -1)
imgDraw = cv2.putText(imgDraw, 'This is a label', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5, cv2.LINE_AA)

stackedImages = stackImages(0.6,([img,imgRotate,imgCropped,imgFlipVert],
                               [imgTranslate,imgDraw,imgRotateFunc,imgFlipHori]))

cv2.imshow("Staked Images", stackedImages)
# cv2.imwrite("assets/new_bg1.jpg", imgRotate)

kernel = np.ones((5,5),np.uint8)
imgInvert = cv2.bitwise_not(img)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img,(19,19),0)
imgCanny = cv2.Canny(img,125,175)
imgDilation = cv2.dilate(imgCanny,kernel , iterations = 2)
imgEroded = cv2.erode(imgDilation,kernel,iterations=2)

stackedImages2 = stackImages(1,([img,imgGray,imgBlur,imgInvert],
                                   [imgCanny,imgDilation,imgEroded,img]))

cv2.imshow("Staked Images 2", stackedImages2)

cv2.waitKey(0)




