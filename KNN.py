import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
                print(f"Add the {j} feature")
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

    timeTaken = time.time() - startTime

    return bestFeatureSet, bestAccuracy, timeTaken

def backwardElimination(xTrain, xTest, yTrain, yTest):

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
                print(f"Remove the {j} feature")
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

    return bestFeatureSet, bestAccuracy
        





def main():
    df = pd.read_csv('C:\\Users\\tejas\\OneDrive\\Documents\\Classes\\AI\\Proj2\\AI_Project_2\\CS170_large_Data__32.txt', delim_whitespace= True, header=None)

    allFeatures = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    # print(allFeatre.columns)

    normalizeFeatures(allFeatures)
    
    

    xTrain, xTest, yTrain, yTest = train_test_split(allFeatures, Y, test_size = 0.3, random_state=42)

    xTrain = pd.DataFrame(xTrain.reset_index(drop=True), columns=allFeatures.columns)
    yTrain = yTrain.reset_index(drop=True)
    xTest = pd.DataFrame(xTest.reset_index(drop=True), columns=allFeatures.columns)

    # print(xTrain.columns)

    bestFeatures, bestAccuracy, timeTaken = forwardSelection(xTrain, xTest, yTrain, yTest)

    # bestFeatures, bestAccuracy = backwardElimination(xTrain, xTest, yTrain, yTest)

    print(f"Accuracy : {bestAccuracy*100:.2f}% for best features : {bestFeatures}")
    print(f"Time taken: {timeTaken}")



        


    
    # print(X.head())
    # print(X.describe())


    

    #Print xtest and ytest to check indices

    
    # print(xTest)
    
        

    

    
    # print(yPred)






if __name__ == '__main__':
    main()