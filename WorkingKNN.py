import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def KNN(xTrain, testRecord, yTrain, k):

    label = None
    distance = float('inf')

    for i in range(xTrain.shape[0]):
        dist = np.sqrt(np.sum(np.square(testRecord - xTrain.iloc[i, :])))

        if(dist < distance):
            distance = dist
            label = yTrain.iloc[i]

    return label
        


def normalizeFeatures(X):

    for feature in X.columns:
        X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()
        X[feature] = X[feature].round(3)



def main():
    df = pd.read_csv('C:\\Users\\tejas\\OneDrive\\Documents\\Classes\\AI\\Proj2\\CS170_small_Data__32.txt', delim_whitespace= True, header=None)
    
    df = df.iloc[:, [0, 1, 3, 5]]

    print(df)

    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    # normalizeFeatures(X)
    
    # print(X.head())
    # print(X.describe())


    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3, random_state=42)

    #Print xtest and ytest to check indices

    xTrain = xTrain.reset_index(drop=True)
    yTrain = yTrain.reset_index(drop=True)
    xTest = xTest.reset_index(drop=True)
    # print(xTest)
    
    accuracy = 0
    for i in range(xTest.shape[0]):
        label = KNN(xTrain, xTest.iloc[i, :], yTrain, 1)
        if label == yTest.iloc[i]:
            accuracy += 1
    
    accuracy /= xTest.shape[0]
    print(accuracy)
        

    

    
    # print(yPred)






if __name__ == '__main__':
    main()