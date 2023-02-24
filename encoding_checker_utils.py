import numpy as np

#####################
# Analyze Encodings #
#####################

# Matrix covariance (assumes matrix is samples x encoding_dim)
def covariance(xArr):
    xMeanArr = np.mean(xArr, axis=0)
    cov = 1 / xArr.shape[0] * np.dot(np.transpose(xArr - xMeanArr), xArr - xMeanArr)
    return cov

# Cross-covariance between matrices (assumes matrix is samples x encoding_dim)
def cross_covariance(xArr1, xArr2):
    # Check this matrix math
    xMeanArr1 = np.mean(xArr1, axis=0)
    xMeanArr2 = np.mean(xArr2, axis=0)
    crossCov = 1 / xArr1.shape[0] * np.dot(np.transpose(xArr1 - xMeanArr1), xArr2 - xMeanArr2)
    return crossCov

# Cross-correlation between matrices (assumes matrix is samples x encoding_dim)
def cross_correlation(xArr1, xArr2):
    crossCor = 1 / xArr1.shape[0] * np.dot(np.transpose(xArr1), xArr2)
    # Barlow Twins crossCor
    #crossCor = np.dot(np.transpose(xArr1), xArr2) / np.dot(np.transpose(np.std(xArr1, axis=0)), np.std(xArr2, axis=0))
    return crossCor

# Get Euclidean distance between an array of vectors and a single vector
def euc_dist_to_one_vec(encArr, encVec):
    distList = []
    for i in range(encArr.shape[0]):
        distList.append(np.linalg.norm(encArr[i, :] - encVec))
    return distList

# Get average Euclidean distance between an array of vectors and a single vector
def avg_euc_dist_to_one_vec(encArr, encVec):
    distList = euc_dist_to_one_vec((encArr, encVec))
    return sum(distList) / len(distList)

# Get average Euclidean distance between every pair of vectors in an array
def avg_euc_dist_between_array(encArr):
    distList = []
    for i in range(encArr.shape[0] - 1):
        for j in range(i + 1, encArr.shape[0]):
            distList.append(np.linalg.norm(encArr[i, :] - encArr[j, :]))
    return sum(distList) / len(distList)

# Angle between two vectors using cosine similarity
def cosine_ang(vec1, vec2):
    cosSim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.arccos(cosSim) * 180 / 3.14159

def cos_ang_to_one_vec(encArr, encVec):
    simList = []
    for i in range(encArr.shape[0]):
        simList.append(cosine_ang(encArr[i, :], encVec))
    return simList

# Get average angle between an array of vectors and a single vector
def avg_cos_ang_to_one_vec(encArr, encVec):
    simList = cos_ang_to_one_vec(encArr, encVec)
    return sum(simList) / len(simList)

# Get average angle between every pair of vectors in an array
def avg_cos_ang_between_array(encArr):
    simList = []
    for i in range(encArr.shape[0] - 1):
        for j in range(i + 1, encArr.shape[0]):
            simList.append(cosine_ang(encArr[i, :], encArr[j, :]))
    return sum(simList) / len(simList)

def one_vec_sparsity(encVec):
    nonZeroBool = np.abs(encVec) >= np.finfo(float).eps
    return nonZeroBool

def avg_sparsity(encArr):
    sparsityList = []
    for i in range(encArr.shape[0]):
        nonzeroBool = np.abs(encArr[i, :]) >= np.finfo(float).eps
        sparsityList.append(sum(nonzeroBool))
    return sum(sparsityList) / len(sparsityList)

def elem_near_zero(encVec):
    nearZeroBool = np.logical_and(np.abs(encVec) >= np.finfo(np.float32).eps,
                                  np.abs(encVec) <= 10 * np.finfo(np.float32).eps)
    return nearZeroBool

def avg_near_zero(encArr):
    nearZeroList = []
    for i in range(encArr.shape[0]):
        nearZeroBool = np.logical_and(np.abs(encArr[i, :]) >= np.finfo(np.float32).eps,
                                      np.abs(encArr[i, :]) <= 10 * np.finfo(np.float32).eps)
        nearZeroList.append(sum(nearZeroBool))
    return sum(nearZeroList) / len(nearZeroList)

##################
# FDA Projection #
##################

def fisher_W(xArr1, xArr2):
    xMeanArr1 = np.mean(xArr1, axis=0)
    xMeanArr2 = np.mean(xArr2, axis=0)
    scatter1 = np.dot(np.transpose(xArr1 - xMeanArr1), xArr1 - xMeanArr1)
    scatter2 = np.dot(np.transpose(xArr2 - xMeanArr2), xArr2 - xMeanArr2)
    totalScatter = scatter1 + scatter2
    W = np.dot(np.linalg.inv(totalScatter), xMeanArr1 - xMeanArr2)
    W = W / np.linalg.norm(W)
    return W