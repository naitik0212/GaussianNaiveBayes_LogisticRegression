import matplotlib.pyplot as plt
import random
import math
from functools import reduce
from pprint import pprint
import numpy as np
import pandas as pd

#Gaussian Naive Bayes Functions
# SplitData for Gaussian Naive Bayes (training and test)
def splitData(lines, n):
    lengthData = len(lines)
    splitsize = int(lengthData / n)
    finalData = [lines[0 + splitsize * i: splitsize * (i + 1)] for i in range(n)]
    leftover = lengthData - splitsize * n
    edge = splitsize * n
    for i in range(leftover):
        finalData[i % n].append(lines[edge + i])

    return finalData

#Selecting the required data
def mergeListsExcept(i, listOfList):
    toReturn = []
    for j in range(len(listOfList)):
        if i != j:
            toReturn.extend(listOfList[j])
    return toReturn


#Splitting based on Class
def splitClass(data):
    dict = {"zero":[],"one":[]}
    for line in data:
        splited = line.split(",")

        if float(splited[-1]) == 0:
            dict["zero"].append(splited)
        else:
            dict["one"].append(splited)
    return dict

#Calculating Mean of each class
def calculateMean(splitClass):
    mean = [0 for i in range(4)]
    meandict = {}
    for classKey in splitClass:
        for i in range(4):
            for j in range(len(splitClass[classKey])):
                mean[i] = mean[i] + float(splitClass[classKey][j][i])
            if len(splitClass[classKey]) != 0:
                mean[i] = mean[i]/(len(splitClass[classKey]))
            else:
                mean[i] = mean[i] / (len(splitClass[classKey])+1)
        meandict[classKey] = mean
        mean = [0 for i in range(4)]

    return meandict

#Calculating Standard Deviation of each Class
def calculateStd(splitClass, calculatedMean):
    stddict = {}

    for classKey in splitClass:
        std = [0 for i in range(4)]
        temp = [0 for i in range(4)]
        for i in range(4):
            for j in range(len(splitClass[classKey])):
                temp[i] = temp[i] + ((float(splitClass[classKey][j][i]) - calculatedMean[classKey][i])**2)
            if(len(splitClass[classKey])!=1):
                std[i] = (temp[i]/(len(splitClass[classKey])-1))**0.5
            else:
                std[i]=(temp[i]/(len(splitClass[classKey])))**0.5
        if(std == 0):
            std = 1
        stddict[classKey] = std
    return stddict

#Calculating tbe number of correctly predicted instances
def calculateProbability(classTest, calculatedMean, calculatedStd):
    correctlypredicted=0
    for classKey in classTest:
        temp1 = [0 for i in range(4)]
        temp2 = [0 for i in range(4)]
        for j in range(len(classTest[classKey])):
            for i in range(4):
                exp1 = [0 for i in range(4)]
                if calculatedStd["zero"][i] == 0:
                    calculatedStd["zero"][i] = 1
                exp1[i] = (((float(classTest[classKey][j][i]) - calculatedMean["zero"][i])**2)/(2*((calculatedStd["zero"][i])**2)))
                temp1[i] = (1 * math.exp(-exp1[i]))/ (((2*math.pi)**0.5) * calculatedStd["zero"][i])
                exp2 = [0 for i in range(4)]
                if calculatedStd["one"][i] == 0:
                    calculatedStd["one"][i] = 1
                exp2[i] = (((float(classTest[classKey][j][i]) - calculatedMean["one"][i]) ** 2) / (2 * ((calculatedStd["one"][i]) ** 2)))
                temp2[i] = (1 * math.exp(-exp2[i])) / (((2 * math.pi) ** 0.5) * calculatedStd["one"][i])

            zero = reduce(lambda x, y: x*y, temp1)
            one = reduce(lambda x, y: x*y, temp2)

            if zero > one:
                if classKey=="zero":
                    correctlypredicted += 1

            if(one > zero) and classKey=="one":
                correctlypredicted += 1
    return correctlypredicted

def finalProbability(trainData,testData, trainPercent):
    random.shuffle(trainData)
    splitDataset = splitData(trainData, trainPercent)
    for i in range(trainPercent):
        if trainPercent == 1:
            trainData = splitDataset[0]
        else:
            trainData = splitDataset[i]

    classTrain = splitClass(trainData)
    classTest = splitClass(testData)
    calculatedMean = calculateMean(classTrain)
    calculatedStd = calculateStd(classTrain,calculatedMean)
    calculatedProbability = calculateProbability(classTest, calculatedMean, calculatedStd)
    return (calculatedProbability / len(testData)) * 100

def generatedSample(calculatedMean, calculatedStd):
    from numpy import array
    a = array(calculatedMean["one"])
    # print(type(a))

    from numpy import array
    b = array(calculatedStd["one"])
    # print(type(b))

    s = np.random.normal(a, b, (400, 4))
    # print(s)
    GeneratedSampleMean1 = np.mean(s[:, 0])
    GeneratedSampleVariance1 = (np.var(s[:, 0]))

    GeneratedSampleMean2 = np.mean(s[:, 1])
    GeneratedSampleVariance2 = np.var(s[:, 1])

    GeneratedSampleMean3 = np.mean(s[:, 2])
    GeneratedSampleVariance3 = np.var(s[:, 2])

    GeneratedSampleMean4 = np.mean(s[:, 3])
    GeneratedSampleVariance4 = np.var(s[:, 3])

    GeneratedSampleMean = []


    GeneratedSampleMean.append(GeneratedSampleMean1)
    GeneratedSampleMean.append(GeneratedSampleMean2)
    GeneratedSampleMean.append(GeneratedSampleMean3)
    GeneratedSampleMean.append(GeneratedSampleMean4)

    GeneratedSampleVariance = []

    GeneratedSampleVariance.append(GeneratedSampleVariance1)
    GeneratedSampleVariance.append(GeneratedSampleVariance2)
    GeneratedSampleVariance.append(GeneratedSampleVariance3)
    GeneratedSampleVariance.append(GeneratedSampleVariance4)

    GeneratedSampleVariance=GeneratedSampleVariance

    # pprint(calculatedMean)
    # pprint(calculatedStd)
    print("Random Sample Mean: ")
    print(GeneratedSampleMean)
    print("Random Sample Variance: ")
    print(GeneratedSampleVariance)


#Liner Regression Functions

def sigmoidfunction(values):
    temp = (1 + np.exp(-1 * values))
    return 1/temp

def partitionTrainData(trainDataPercent,train,trainClass):
    splitLR = np.random.rand(len(train)) < trainDataPercent
    partitionedtrain = train[splitLR]
    partitionedtrainClass = trainClass[splitLR]
    return partitionedtrain,partitionedtrainClass

def difference(trainClass,predictedClassLR):
    return trainClass - predictedClassLR

def dotProduct(a,b):
    return np.dot(a,b)

def logisticRegressionmodel(train, trainClass):
    weights = np.zeros(train.shape[1])
    learnrate = 0.0001
    epoch = 1000
    for i in range(epoch):
        values = dotProduct(train, weights)
        predictedClassLR = sigmoidfunction(values)
        dif = difference(trainClass,predictedClassLR)
        grad = dotProduct(train.T, dif)
        weights = weights + (learnrate * grad)
    return weights

def testPrediction(test, testClass, trainWeights):
    testWeights=dotProduct(test, trainWeights)
    testprediction = sigmoidfunction(testWeights)
    finalpredictedclass = (np.round(testprediction))
    correctlyTested=((finalpredictedclass == testClass).sum())
    return correctlyTested*100/len(testClass)

#GraphPlot for both Gaussian Naive Bayes and Logistic Regression

def graphPlot(graph1,graph2,trainDataPercent):
    plt.plot(graph1, label="Gausian Naive Bayes")
    plt.plot(graph2, label="Logistic Regression")
    xdatapoints=[i for i in range(0, len(trainDataPercent))]
    plt.ylabel('accuracy')
    plt.xlabel('Datapoints')
    plt.xticks(xdatapoints,trainDataPercent)
    plt.legend()
    plt.show()

def main():
    #Gausian Naive Bayes
    text_file = open("Bank_Data", "r")
    lines2 = pd.read_csv('Bank_Data', header=None, sep=",")
    lines = text_file.readlines()
    random.shuffle(lines)
    splitDataset = splitData(lines, 3)
    for i in range(3):
            testData = splitDataset[i]
            trainData = mergeListsExcept(i, splitDataset)

    trainDataPercent = [0.01, 0.02, 0.05, 0.1, 0.625, 1]

    graph1 = []
    for percent in trainDataPercent:
        temp = []
        for i in range(5):
            trainpercent = int(1 / percent)
            temp.append(finalProbability(trainData, testData, trainpercent))
        graph1.append(sum(int(v) for v in temp)/5)
    print("Average Accuracy for Gaussian Naive Bayes over 5 runs with varied training data")
    print(graph1)

    #Logistic Regression
    #Used NumPy and Pandas
    lines2.sample(frac=1)
    initialData = np.array(lines2.iloc[:, 0:4])
    initialClass = np.array(lines2.iloc[:, 4])
    splitLR = np.random.rand(len(initialData)) < (2/3)
    train = initialData[splitLR]
    trainClass = initialClass[splitLR]
    test = initialData[~splitLR]
    testClass = initialClass[~splitLR]

    graph2 = []
    for percent in trainDataPercent:
        temp = []
        for i in range(5):
            trainPercent = percent
            partitionedtrain, partitionedtrainClass = partitionTrainData(trainPercent, train, trainClass)
            trainWeights = logisticRegressionmodel(partitionedtrain, partitionedtrainClass)
            temp.append(testPrediction(test, testClass, trainWeights))
            # print(temp)
        graph2.append(sum(float(v) for v in temp) / 5)
    print("Average Accuracy for Logistic Regression over 5 runs with varied training data")
    print(graph2)


    # Part 3
    nooffolds = 3
    random.shuffle(lines)
    splitDataset = splitData(lines, nooffolds)

    for i in range(nooffolds):
        if nooffolds == 1:
            trainData = splitDataset[0]
            testData = splitDataset[0]
        else:
            trainData = mergeListsExcept(i, splitDataset)
            testData = splitDataset[i]

        classTrain = splitClass(trainData)
        classTest = splitClass(testData)

        calculatedMean = (calculateMean(classTrain))

        print("Mean of Model {} is".format(i+1))

        print(calculatedMean["one"])

        calculatedStd = (calculateStd(classTrain, calculatedMean))

        print("Variance of Model {} is".format(i + 1))
        finalVarianceModel= ([i ** 2 for i in calculatedStd["one"]])
        print(finalVarianceModel)

        Accuracy = finalProbability(trainData, testData, 1)
        print("Accuracy of Model {}: is {}".format(i+1, Accuracy))

    generatedSample(calculatedMean, calculatedStd)
    graphPlot(graph1, graph2,trainDataPercent)


if __name__ == "__main__" :
    main()