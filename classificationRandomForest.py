import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import random
from sklearn.ensemble import RandomForestClassifier

paths = ["images/raw", "images/learn"]
images = []
done = []
raw = []

#read images from files
def readImages():
    for path in paths:
        for r, d, f in os.walk(path):
            for file in f:
                print(file)
                if path == "images/raw":
                    images.append(cv.imread(os.path.join(r, file), 1))
                    raw.append(cv.imread(os.path.join(r, file), 1))
                else:
                    done.append(cv.imread(os.path.join(r, file), 1))

#show images
def showImages(images, number, rows, columns):
    plt.figure(figsize=(20, 20))
    for i in range(1, number + 1):
        plt.subplot(rows, columns, i)
        plt.imshow(images[i - 1][:, :, ::-1]) #rgb
    plt.show()
    plt.close()
    plt.clf()

def image_colorfulness(image):
	(B, G, R) = cv.split(image.astype("float"))
	rg = np.absolute(R - G)
	yb = np.absolute(0.5 * (R + G) - B)
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	return stdRoot + (0.3 * meanRoot)

def calculatePropertiesAndReturn(sample):
    variance = image_colorfulness(sample);
    sample = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
    moments = cv.moments(sample)
    huMoments = cv.HuMoments(moments)
    return ([variance,
                        moments["m00"],
                        moments["m10"],
                        moments["m01"],
                        moments["m20"],
                        moments["m02"],
                        moments["m11"],
                        moments["m30"],
                        moments["m21"],
                        moments["m12"],
                        moments["m03"],
                        moments["mu20"],
                        moments["mu11"],
                        moments["mu30"],
                        moments["mu21"],
                        moments["mu12"],
                        moments["mu03"],
                        moments["nu20"],
                        moments["nu11"],
                        moments["nu02"],
                        moments["nu30"],
                        moments["nu21"],
                        moments["nu12"],
                        moments["nu03"],
                        huMoments[0][0], huMoments[1][0], huMoments[2][0], huMoments[3][0], huMoments[4][0],
                        huMoments[5][0], huMoments[6][0]])




def calculateProperties(sample, id, i, j):
    variance = image_colorfulness(sample);
    sample = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
    moments = cv.moments(sample)
    huMoments = cv.HuMoments(moments)
    learningSet.append([variance,
                        moments["m00"],
                        moments["m10"],
                        moments["m01"],
                        moments["m20"],
                        moments["m02"],
                        moments["m11"],
                        moments["m30"],
                        moments["m21"],
                        moments["m12"],
                        moments["m03"],
                        moments["mu20"],
                        moments["mu11"],
                        moments["mu30"],
                        moments["mu21"],
                        moments["mu12"],
                        moments["mu03"],
                        moments["nu20"],
                        moments["nu11"],
                        moments["nu02"],
                        moments["nu30"],
                        moments["nu21"],
                        moments["nu12"],
                        moments["nu03"],
                        huMoments[0][0], huMoments[1][0], huMoments[2][0], huMoments[3][0], huMoments[4][0], huMoments[5][0], huMoments[6][0]])
    decisions.append(done[id][i][j][0])

readImages()

learningSet = []
decisions=[]

#Wybór obrazu [0] - zdjęcie to będzie poddane testowaniu, a na pozostałych 5 algorytm będzie się uczył
images[0], images[2] = images[2], images[0]
done[0], done[2] = done[2], done[0]

#Tworzenie zbioru uczącego
for iter in range(50000):
    id = random.randint(1, 5)
    i = random.randint(3, len(images[id])-3)
    j = random.randint(3, len(images[id][i])-3)
    while(done[id][i][j][0]!=255):
        i = random.randint(3, len(images[id])-3)
        j = random.randint(3, len(images[id][i])-3)

    sample = images[id][i-2:i+3, j-2:j+3]

    if len(sample) == 5:
                calculateProperties(sample, id, i, j)

    i = random.randint(3, len(images[id]) - 3)
    j = random.randint(3, len(images[id][i]) - 3)
    while (done[id][i][j][0] != 0):
        i = random.randint(3, len(images[id]) - 3)
        j = random.randint(3, len(images[id][i]) - 3)

    sample = images[id][i - 2:i + 3, j - 2:j + 3]

    if len(sample) == 5:
        calculateProperties(sample, id, i, j)

#Inicjalizacja klasyfikatora
knn = RandomForestClassifier()
knn.fit(learningSet, decisions)

imageCopy = images[1].copy()
q=0
accuracy = 0
sensitivity = 0
specificity = 0
id = 0
tp=0
fp=0
tn=0
fn=0
for i in range(3, len(images[id])-3):
    print(i)
    p=[]
    for j in range(3, len(images[id][i])-3):
        real = done[id][i][j][0]
        sample = images[id][i - 2:i + 3, j - 2:j + 3]
        properties = calculatePropertiesAndReturn(sample)
        p.append(properties)

    j=3
    predictions = knn.predict(p)
    for prediction in predictions:
        imageCopy[i][j] = prediction
        q +=1
        if done[id][i][j][0] == prediction:
                    accuracy += 1
        if (prediction == 255 and done[id][i][j][0] == 255) or (prediction == 0 and done[id][i][j][0] == 255):
                    sensitivity += 1
        if (prediction == 255 and done[id][i][j][0] == 0) or (prediction == 0 and done[id][i][j][0] == 0):
                    specificity += 1
        if (prediction == 255 and done[id][i][j][0] == 255):
            tp+=1
        if (prediction == 0 and done[id][i][j][0] == 255):
            fn+=1
        if (prediction == 0 and done[id][i][j][0] == 0):
            tn+=1
        if (prediction == 255 and done[id][i][j][0] == 0):
            fp+=1

        j+=1

accuracy /= q
sensitivity = tp / sensitivity
specificity = tn / specificity
mcc = (tp*tn - fp*fn)/(((tp+fp)*(fn+tn)*(fp+tn)*(tp+fn))**(1/2))
print("Accuracy: ", accuracy)
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
print("Arithmetic mean: ", (sensitivity + specificity)/2)
print("Geometric mean: ", (sensitivity * specificity)**(0.5))
print("MCC: ", mcc)
showImages([images[id], done[id], imageCopy], 3, 1, 3)


del images
del learningSet
del raw
del done
