import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import time

def KNN(xTrain, testRecord, yTrain):
    #print(testRecord)
    label = None
    distance = float('inf')

    for i in range(xTrain.shape[0]):
        #print(testRecord, xTrain.values)
        dist = np.sqrt(np.sum(np.square(testRecord.values - xTrain.iloc[i].values)))

        if(dist < distance):
            distance = dist
            label = yTrain.iloc[i]

    return label
        

def callKNN(xTrain, xTest, yTrain, yTest, featureSet):

    
    # print(featureSet)
    accuracy = 0
    for i in range(xTest.shape[0]):
        label = KNN(xTrain.iloc[:, list(f - 1 for f in featureSet)], xTest.iloc[i, [f - 1 for f in featureSet]], yTrain)
        if label == yTest.iloc[i]:
            accuracy += 1
    
    accuracy /= xTest.shape[0]

    return accuracy


def normalizeFeatures(allFeatures):
    for feature in allFeatures.columns:
        allFeatures[feature] = (allFeatures[feature] - allFeatures[feature].mean()) / allFeatures[feature].std()
        allFeatures[feature] = allFeatures[feature].round(3)


def forwardSelection(xTrain, xTest, yTrain, yTest):

    reducedAccuracyCounter = 0

    startTime = time.time()

    currentFeatureSet = set()
    bestFeatureSet = set()
    bestAccuracy = 0

    for i in range(1, xTrain.shape[1] + 1):
        print(f"We are on level {i} of the tree")
        currentFeature = None
        currentAccuracy = 0

        for j in range(1, xTrain.shape[1] + 1):
            if j not in currentFeatureSet:
                featuresTemp = set(currentFeatureSet)   #Temp copy
                featuresTemp.add(j)     #Add the j'th feature
                print(f"Add feature {j}")
                accuracy = callKNN(xTrain, xTest, yTrain, yTest, featuresTemp)
                if accuracy > currentAccuracy:
                    currentAccuracy = accuracy
                    currentFeature = j
        
        currentFeatureSet.add(currentFeature)
        print(f"feature added : {currentFeature}")
        print(f"accuracy for current features: {currentFeatureSet} is {currentAccuracy*100:.2f} %")

        if bestAccuracy < currentAccuracy:
            print(f"accuracy improvement = {(currentAccuracy - bestAccuracy)*100:.2f}%")
            bestAccuracy = currentAccuracy
            bestFeatureSet = set(currentFeatureSet)
        else:
            if reducedAccuracyCounter < 3:
                reducedAccuracyCounter += 1
            else:
                timeTaken = time.time() - startTime
                return bestFeatureSet, bestAccuracy, timeTaken

    timeTaken = time.time() - startTime

    return bestFeatureSet, bestAccuracy, timeTaken

def backwardElimination(xTrain, xTest, yTrain, yTest):

    startTime = time.time()

    currentFeatureSet = set(xTrain.columns)
    bestFeatureSet = set()
    bestAccuracy = callKNN(xTrain, xTest, yTrain, yTest, currentFeatureSet)
    print(f"Accuracy using {currentFeatureSet} : {bestAccuracy*100:.2f}")

    for i in range(1, xTrain.shape[1] + 1):
        print(f"We are on level {i} of the tree")
        currentFeature = None
        currentAccuracy = 0

        for j in range(1, xTrain.shape[1] + 1):
            if j in currentFeatureSet:
                featuresTemp = set(currentFeatureSet)   #Temp copy
                featuresTemp.remove(j)     #Add the j'th feature
                print(f"Remove feature {j}")
                accuracy = callKNN(xTrain, xTest, yTrain, yTest, featuresTemp)
                if accuracy > currentAccuracy:
                    currentAccuracy = accuracy
                    currentFeature = j
        
        currentFeatureSet.remove(currentFeature)
        print(f"feature removed : {currentFeature}")
        print(f"accuracy for current features: {currentFeatureSet} is {currentAccuracy*100:.2f} %")

        if bestAccuracy < currentAccuracy:
            print(f"accuracy improvement = {(currentAccuracy - bestAccuracy)*100:.2f}%")
            bestAccuracy = currentAccuracy
            bestFeatureSet = set(currentFeatureSet)

    timeTaken = time.time() - startTime

    return bestFeatureSet, bestAccuracy, timeTaken
        





def main():

    print("Welcome to Tejas Deshpande's Feature Selection Algorithm.")
    inputFile = input('Type in the name of the file to test: ')

    if 'csv' in inputFile:
        df = pd.read_csv(inputFile, header=None)
    else:
        df = pd.read_csv(inputFile, delim_whitespace= True, header=None)

    algorithmNumber = int(input('Type the number of the algorithm you want to run.\n 1) Forward Selection \n 2) Backward Elimination \n'))

    randomSampled = int(input('Do you want to run the search on reduced randomly sampled data for faster results ? Enter the number of the choice \n 1) Yes \n 2) No \n'))


    # df = pd.read_csv('CS170_small_Data__8.txt', delim_whitespace= True, header=None)
    # df = pd.read_csv('CS170_large_Data__8.txt', delim_whitespace= True, header=None)
    # df = pd.read_csv('CS170_XXXlarge_Data__8.txt', delim_whitespace= True, header=None)

    if randomSampled == 1:
        print(f"Dataset Shape before random sampling: {df.shape}")
        df = resample(df, n_samples=int(df.shape[0]/2))
        print(f"Dataset Shape after random sampling: {df.shape}")

    allFeatures = df.iloc[:, 1:]
    Y = df.iloc[:, 0]
    normalizeFeatures(allFeatures)
    
    

    xTrain, xTest, yTrain, yTest = train_test_split(allFeatures, Y, test_size = 0.3, random_state=42)

    xTrain = pd.DataFrame(xTrain.reset_index(drop=True), columns=allFeatures.columns)
    yTrain = yTrain.reset_index(drop=True)
    xTest = pd.DataFrame(xTest.reset_index(drop=True), columns=allFeatures.columns)

    if algorithmNumber == 1:
        bestFeatures, bestAccuracy, timeTaken = forwardSelection(xTrain, xTest, yTrain, yTest)
        print(f"Accuracy : {bestAccuracy*100:.2f}% for best features : {bestFeatures}")
        print(f"Time taken: {(timeTaken/60):.2f} minutes")
        print("Forward Selection Search complete")

    else:
        bestFeatures, bestAccuracy, timeTaken = backwardElimination(xTrain, xTest, yTrain, yTest)
        print(f"Accuracy : {bestAccuracy*100:.2f}% for best features : {bestFeatures}")
        print(f"Time taken: {(timeTaken/60):.2f} minutes")
        print("Backward Elimination Search complete")






if __name__ == '__main__':
    main()