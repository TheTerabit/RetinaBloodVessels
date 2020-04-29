import cv2 as cv
from matplotlib import pyplot as plt
import os
import tranformations as t #plik nagłówkowy

paths = ["images/raw", "images/learn"]

images = []
done = []
raws = []

#read images from files
def readImages():
    for path in paths:
        for r, d, f in os.walk(path):
            for file in f:
                print(file)
                if path == "images/raw":
                    images.append(cv.imread(os.path.join(r, file), 1))
                    raws.append(cv.imread(os.path.join(r, file), 1))
                else:
                    done.append(cv.imread(os.path.join(r, file), 1))

#show images
def showImages(images, number, rows, columns):
    plt.figure(figsize=(20, 20))
    for i in range(1, number + 1):
        plt.subplot(rows, columns, i)
        plt.imshow(images[i - 1], cmap = "gray") #bgr
    plt.show()
    plt.close()
    plt.clf()


def greenFilter(image):
    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j]=[0, image[i][j][1], 0]
    return image


#main transformation
def transformImages(images):
    for i in range(len(images)):
        print("Start of tranformations...")
        blurTemp = t.blur(images[i], 5, 5)
        greenTemp = greenFilter(blurTemp)
        grayTemp = t.bgr2gray(greenTemp)
        gammaTemp = t.gamma(grayTemp, 3)
        edgesTemp = cv.Canny(gammaTemp, 10, 30 ,3)
        dilationTemp = t.dilation(edgesTemp, 1)
        erosionTemp = t.erosion(dilationTemp, 1)
        images[i] = erosionTemp


readImages()
transformImages(images)
print("Showing images...")
for raw, learn in zip(images, done):
    q = len(raw) * len(raw[0])
    accuracy = 0
    sensitivity = 0
    specificity = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(raw)):
        for j in range(len(raw[i])):
            if learn[i][j][0] == raw[i][j]:
                accuracy += 1
            if (raw[i][j] == 255 and learn[i][j][0]==255) or (raw[i][j] == 0 and learn[i][j][0]==255):
                sensitivity += 1
            if (raw[i][j] == 255 and learn[i][j][0]==0) or (raw[i][j] == 0 and learn[i][j][0]==0):
                specificity += 1
            if (raw[i][j] == 255 and learn[i][j][0]==255):
                tp += 1
            if (raw[i][j] == 0 and learn[i][j][0]==255):
                fn += 1
            if (raw[i][j] == 0 and learn[i][j][0]==0):
                tn += 1
            if (raw[i][j] == 255 and learn[i][j][0]==0):
                fp += 1
    accuracy /= q
    sensitivity = tp / sensitivity
    specificity = tn / specificity
    mcc = (tp * tn - fp * fn) / (((tp + fp) * (fn + tn) * (fp + tn) * (tp + fn)) ** (1 / 2))
    print("Accuracy: ", accuracy)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Arithmetic mean: ", (sensitivity + specificity) / 2)
    print("Geometric mean: ", (sensitivity * specificity) ** (0.5))
    print("MCC: ", mcc)

    showImages([raws[0], learn, raw], 3, 1, 3)


del images
del done
del raw
