import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import time

# K-Nearest Neighbour Classifier
def KNN(xTrain, testRecord, yTrain):
    label = None
    distance = float('inf')

    for i in range(xTrain.shape[0]):
        dist = np.sqrt(np.sum(np.square(testRecord.values - xTrain.iloc[i].values)))       #Calculate Euclidean distance
        if(dist < distance):
            distance = dist
            label = yTrain.iloc[i]      #Update test sample label for nearest neighbour

    return label
        
# Method to call KNN Classifier for set of features selected by Forward Selection and Backward Elimination algorithms
# for testing the accuracy of different feature sets.

def callKNN(xTrain, xTest, yTrain, yTest, featureSet):

    accuracy = 0
    for i in range(xTest.shape[0]):         #For each test sample call KNN with the feature set selected by feature search algorithms
        label = KNN(xTrain.iloc[:, list(f - 1 for f in featureSet)], xTest.iloc[i, [f - 1 for f in featureSet]], yTrain)
        if label == yTest.iloc[i]:
            accuracy += 1
    
    accuracy /= xTest.shape[0]              #Calculate accuracy of KNN for entire test set for the selected features

    return accuracy

# Method to normalize input dataset features 
def normalizeFeatures(allFeatures):
    for feature in allFeatures.columns:
        allFeatures[feature] = (allFeatures[feature] - allFeatures[feature].mean()) / allFeatures[feature].std()
        allFeatures[feature] = allFeatures[feature].round(3)

#Method for Forward Selection
def forwardSelection(xTrain, xTest, yTrain, yTest):

    reducedAccuracyCounter = 0      #Counter to keep track of number of times the accuracy drops for Anytime Optimization

    startTime = time.time()

    currentFeatureSet = set()       #Set to keep track of best features in each iteration or each level of the search tree
    bestFeatureSet = set()          #Set to keep track of the subset of best features for classification found so far
    bestAccuracy = 0                #Keeping track of the accuracy for best features for classification found so far

    for i in range(1, xTrain.shape[1] + 1):
        print(f"We are on level {i} of the tree")
        currentFeature = None       # Feature to be added at current level of search tree which shows max increase in accuracy.
        currentAccuracy = 0         # Keeping track of the max accuracy given by the best feature for the current iteration / level of search tree.

        for j in range(1, xTrain.shape[1] + 1):
            if j not in currentFeatureSet:
                featuresTemp = set(currentFeatureSet)   #Temp copy to find the best feature(child node) at current level of the search tree along with previously selected features
                featuresTemp.add(j)     #Add the j'th feature
                print(f"Add feature {j}")
                accuracy = callKNN(xTrain, xTest, yTrain, yTest, featuresTemp)      #Call KNN to test the set of currently selected features.
                if accuracy > currentAccuracy:                    
                    currentAccuracy = accuracy                      #Update accuracy for the feature selected at current level of the search tree.
                    currentFeature = j                              #Update best current feature if accuracy increases.
        
        currentFeatureSet.add(currentFeature)
        print(f"feature added : {currentFeature}")
        print(f"accuracy for current features: {currentFeatureSet} is {currentAccuracy*100:.2f} %")

        #Update the best accuracy and best set of classification features found so far.
        if bestAccuracy < currentAccuracy:
            print(f"accuracy improvement = {(currentAccuracy - bestAccuracy)*100:.2f}%")
            bestAccuracy = currentAccuracy
            bestFeatureSet = set(currentFeatureSet)
        else:
            if reducedAccuracyCounter < 3:     #Keep track of number of times the accuracy decreased for the current search
                reducedAccuracyCounter += 1
            else:
                timeTaken = time.time() - startTime
                return bestFeatureSet, bestAccuracy, timeTaken          #stop if accuracy decreases more than 3 times during feature search

    timeTaken = time.time() - startTime

    return bestFeatureSet, bestAccuracy, timeTaken         #Return the best set of features and best accuracy along with time taken for the entire search

def backwardElimination(xTrain, xTest, yTrain, yTest):

    startTime = time.time()

    currentFeatureSet = set(xTrain.columns)         #Set to keep track of best features in each iteration or each level of the search tree
    bestFeatureSet = set()                          #Set to keep track of the subset of best features for classification found so far
    bestAccuracy = callKNN(xTrain, xTest, yTrain, yTest, currentFeatureSet)     #Keeping track of the accuracy for best features for classification found so far
    print(f"Accuracy using {currentFeatureSet} : {bestAccuracy*100:.2f}")

    for i in range(1, xTrain.shape[1] + 1):
        print(f"We are on level {i} of the tree")
        currentFeature = None          # Feature to be removed at current level of search tree which shows max increase in accuracy.
        currentAccuracy = 0            # Keeping track of the max accuracy given by removing the worst feature for the current iteration / level of search tree. 

        for j in range(1, xTrain.shape[1] + 1):
            if j in currentFeatureSet:
                featuresTemp = set(currentFeatureSet)   #Temp copy to find the worst feature(child node) to be removed at current level of the search tree along with previously removed features
                featuresTemp.remove(j)     #Remove the j'th feature
                print(f"Remove feature {j}")
                accuracy = callKNN(xTrain, xTest, yTrain, yTest, featuresTemp)      #Call KNN to test the set of currently selected features.
                if accuracy > currentAccuracy:
                    currentAccuracy = accuracy               #Update accuracy for the feature removed at current level of the search tree.
                    currentFeature = j                       #Update worst current feature if accuracy increases.
        
        currentFeatureSet.remove(currentFeature)             #Remove the worst feature found at this level of the search tree
        print(f"feature removed : {currentFeature}")
        print(f"accuracy for current features: {currentFeatureSet} is {currentAccuracy*100:.2f} %")

        #Update the best accuracy and best set of classification features found so far.
        if bestAccuracy < currentAccuracy:
            print(f"accuracy improvement = {(currentAccuracy - bestAccuracy)*100:.2f}%")
            bestAccuracy = currentAccuracy
            bestFeatureSet = set(currentFeatureSet)

    timeTaken = time.time() - startTime

    return bestFeatureSet, bestAccuracy, timeTaken      #Return the best set of features and best accuracy along with time taken for the entire search
        

def main():

    #Take input from the user
    print("Welcome to Tejas Deshpande's Feature Selection Algorithm.")
    inputFile = input('Type in the name of the file to test: ')

    #Parse the input dataset file
    if 'csv' in inputFile:
        df = pd.read_csv(inputFile, header=None)
    else:
        df = pd.read_csv(inputFile, delim_whitespace= True, header=None)

    algorithmNumber = int(input('Type the number of the algorithm you want to run.\n 1) Forward Selection \n 2) Backward Elimination \n'))

    #Choice for the user to run the algorithm for reduced dataset size for ending the feature search quickly.
    randomSampled = int(input('Do you want to run the search on reduced randomly sampled data for faster results ? Enter the number of the choice \n 1) Yes \n 2) No \n'))

    if randomSampled == 1:
        print(f"Dataset Shape before random sampling: {df.shape}")
        df = resample(df, n_samples=int(df.shape[0]/2))
        print(f"Dataset Shape after random sampling: {df.shape}")

    #Separate the features from class lables
    allFeatures = df.iloc[:, 1:]  
    Y = df.iloc[:, 0]

    #Normalize the features
    normalizeFeatures(allFeatures)   
    
    
    #Split the input dataset into training and testing data samples
    xTrain, xTest, yTrain, yTest = train_test_split(allFeatures, Y, test_size = 0.3, random_state=42)
    xTrain = pd.DataFrame(xTrain.reset_index(drop=True), columns=allFeatures.columns)
    yTrain = yTrain.reset_index(drop=True)
    xTest = pd.DataFrame(xTest.reset_index(drop=True), columns=allFeatures.columns)

    #Call the feature search algorithm selected by the user.
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