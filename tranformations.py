#nasz plik nagłówkowy z funkcjami służącymi do przetwarzania obrazu
import cv2 as cv
import numpy as np
from scipy import ndimage as ndi

MAXIMUM_SIZE_CONTOUR = 250000
MINIMUM_SIZE_CONTOUR = 20000
claheObj = cv.createCLAHE(clipLimit = 20.0, tileGridSize=(10, 10)) #creates clahe object

def gamma(image, gamma = 1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv.LUT(image, table)

def bgr2gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def contrast(image, perc):
    MIN = np.percentile(image, perc)
    MAX = np.percentile(image, 100-perc)
    image = (image - MIN) / (MAX - MIN)
    image[image[:,:] > 255] = 255
    image[image[:,:] < 0] = 0
    image = image.astype(np.uint8)
    return image

def dilation(image, i):
    kernel = np.ones((5, 5), np.uint8)
    return cv.dilate(image, kernel, iterations = i)

def erosion(image, i):
    kernel = np.ones((5, 5), np.uint8)
    return cv.erode(image, kernel, iterations = i)

def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

def closing(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

def laplacian(image):
    return cv.Laplacian(image, cv.CV_64F)

def adaptiveThresh(image):
    #return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    blur = cv.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

def thresh(image, value):
    ret, thresh1 = cv.threshold(image, value, 255, cv.THRESH_BINARY)
    return thresh1


def contours(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def blur(image, x, y):
    return cv.blur(image, (x, y))


def biBlur(image, x, y, z):
    return cv.bilateralFilter(image, x, y, z)


def clahe(image):
    return claheObj.apply(image)


def canny(image):
    return cv.Canny(image, 100, 200)


def gradient(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)


def fillHoles(image):#0-1 rgb
    return ndi.binary_fill_holes(image)


def hull(contour):
    return cv.convexHull(contour)


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def crop(contour):
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    #if image is rotated send coordinates reversed
    if(distance(box[0], box[1]) < distance(box[0], box[3])): return [box[3], box[0], box[1], box[2], rect[1]]
    return [box[0], box[1], box[2], box[3], rect[1][::-1]]
