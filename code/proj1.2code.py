
# coding: utf-8

# In[376]:


#importing packages
# A package is a collection/directory of python modules
from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[377]:


# setting up parameters
# here M is number of basis function and lambda is regularizer
maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False


# In[378]:


# There are 2 solutions for the given problem, one is closed form solution and another is gradient decent
# Opening and reading the csv file 
# the csv file includes target value 
# csv file is a file with comma separated values
# Converting the csv file into list of 69623 targets
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    print("Raw Training Generated..")
    return t

# Opening and reading the csv file 
# the csv file includes 46 features values for each input value(x)
# Converting it into an array of dimentions (69623,46)
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   

# deleating 5 feature columns from 46 feature columns because these columns have zero varience and will not impact our model.
# hence, the final dataMatrix is of dimentions (41,69623)
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)     
    print ("Data Matrix Generated..")
    return dataMatrix

# The target and the input data is to be divided into 3 data sets for training, validation and testing
# Selecting 80% rows for raw training data for the training target
# Final dimentions would be 55699 target values
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    print(str(TrainingPercent) + "% Training Target Generated..")
    return t

# Getting 80% rows from raw data, which would be our training data
# Final dimentions of the array would be (41,55699)
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    # here, slicing of the raw data is done
    d2 = rawData[:,0:T_len]
    print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# Generating array of validation data
# The validation data is 10% of the total length
# Final dimentions of array is (41,6962)
# val size is 6963 (10% of raw data)
# V_end for validation data would be row 62662 (55699+6963)
# Testin data would be from end of testing data to 69624 row
# Start of validation data is 55699 and end is 62662
# Start of testing data is 62661 and end is 69624
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount:V_End-1]
    print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

# generating a validation target value list (10% of total length)
# final dimentions of list would be 6962
# start of validation data is 55699 and end is 62662
# start of testing data is 62661 and end is 69624
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount:V_End-1]
    print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# Big sigma is called Covarience Matrix
# the covarience matrix has variences along the digonal and all other places are set to zero because we need not to compare varience of one feature value with the other feature value.
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    # returns a new array of dimentions(41,41), filled with zeros
    BigSigma    = np.zeros((len(Data),len(Data)))
     # transposing the data matrix and calculating the training length
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    # We multiply the covarience matrix by 200 to increase the width of the basis function to get better approximation 
    # and hence to get better results because it is the covarience matrix that decides how broadly basis function spreads.
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    print ("BigSigma Generated..")
    return BigSigma

# this is a step in calculating gausian basis function. Mathematically, this calculation is a part of gausian basis function formula
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

# calculating the value of gausian basis function here
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# here the final phi matrix is calculated
# Dimentions of raw data is (41,69623)
# Dimentions of Mu matrix is (10,41)
# Dimentions of Big Sigma is (41,41)
# Training percent is 80%
# Dimentions of phi matrix is (55699,10)
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    print ("PHI Generated..")
    return PHI

# finally we are calculating the weights here
# Dimentions of phi matrix is (55699,10)
# Dimentions of target is (55699,1)
# PHI_T is the transpose of the phi matrix
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    print ("Training Weights Generated..")
    return W

# Generation of phi matrix
# here, the dimentions of mu matrix is (10,41)
# the dimentions of Big Sigma is (10,41)
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    print ("Test Out Generated..")
    return Y

# calculating error root mean square
# Dimention of Erms is (55699,1)
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    print ("Accuracy Generated..")
    print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[379]:


# fetching raw data and raw target
RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)


# ## Prepare Training Data

# In[380]:


# passing the raw data and training percent to get the length of training target and training data
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[381]:


# The validation data is used for frequent evaluation of our learning model
# Passing the length of training target to get the length of validation list/array.
# Validation data will start at the end of training data
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[382]:


# Passing the length of training and validation data to get the length of test list/array.
# Test data will start at the end of validation data
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[383]:


ErmsArr = []
AccuracyArr = []

# k means is a stochastic so to make it deterministic you nedd to make random state 0
# we need to find kmeans for training data
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
# Mean(Mu) foe each cluster is the centroid of each cluster
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[384]:


# printing dimentions of mu matrix, covarience matrix, Training phi martix, weights matrix, validation phi matrix and test phi matrix
print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[385]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

# Calculating the root means square error for training, validation and test data
# The root mean square error is used to measure the difference between values predicted by a model and values observed
TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[386]:


print ('UBITname      = karanman')
print ('Person Number = 50290755')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 10 \nLambda = 0.93")
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("Accuracy Training   = " + str(float(TrainingAccuracy.split(',')[0])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("Accuracy Validation = " + str(float(ValidationAccuracy.split(',')[0])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
print ("Accuracy Testing    = " + str(float(TestAccuracy.split(',')[0])))


# ## Gradient Descent solution for Linear Regression

# In[420]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[421]:


# here we are randomly initializing weights and other hyperparameters
# In gradient decent solution weights are updated iteratively
# Learning Rate decides how big each upadte step would be
W_Now        = np.dot(220, W)
La           = 2
learningRate = 0.01
L_Erms_Val   = []
L_Accu_Val   = []
L_Erms_TR    = []
L_Accu_TR    = []
L_Erms_Test  = []
L_Accu_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    # calculating delta E_D by partially  differentiating E with respect to w1
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    # calculating lambda_delta_E_W
    # Lambda is regularization cofficient here. It is a hyperparameter
    La_Delta_E_W  = np.dot(La,W_Now)
    # Adding delta E_D and lambda_delta_E_W to get Delta_E
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)  
    # here the weights are updated by multiplication of learning rate and Delta_E
    # The new weights are weight_now plus Delat_W
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    # weight_next is assigned to Weight_now for the next iteration
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    # Calculating the root means square error for training data
    # The root mean square error is used to measure the difference between values predicted by a model and values observed
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    # Calculating the root means square error for validation data
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    # Calculating the root means square error for testing data
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[422]:


print ('----------Gradient Descent Solution--------------------')
print ("nLambda  = 2\neta=0.01")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

