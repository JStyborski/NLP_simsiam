import numpy as np

#####################
# Analyze Encodings #
#####################

# Cross-covariance between matrices (assumes encoding_dim x samples)
def cross_cov(xArr1, xArr2):
    xMeanArr1 = np.mean(xArr1, axis=1, keepdims=True)
    xMeanArr2 = np.mean(xArr2, axis=1, keepdims=True)
    crossCov = 1 / xArr1.shape[1] * np.dot(xArr1 - xMeanArr1, np.transpose(xArr2 - xMeanArr2))
    return crossCov

# Cross-correlation between matrices (assumes encoding_dim x samples)
def cross_corr(xArr1, xArr2):
    crossCor = 1 / xArr1.shape[1] * np.dot(xArr1, np.transpose(xArr2))
    # Barlow Twins crossCor
    # crossCor = np.dot(np.transpose(xArr1), xArr2) / np.dot(np.transpose(np.std(xArr1, axis=0)), np.std(xArr2, axis=0))
    return crossCor

# Get list of Euclidean distances between an array of vectors and a single vector (assumes encoding_dim x samples)
def euc_dist_to_one_vec(encArr, encVec):
    distList = []
    for i in range(encArr.shape[1]):
        distList.append(np.linalg.norm(encArr[:, i] - encVec))
    return distList

# Get average Euclidean distance between an array of vectors and a single vector (assumes encoding_dim x samples)
def avg_euc_dist_to_one_vec(encArr, encVec):
    distList = euc_dist_to_one_vec(encArr, encVec)
    return sum(distList) / len(distList)

# Get average Euclidean distance between every pair of vectors in an array (assumes encoding_dim x samples)
def avg_euc_dist_bt_array(encArr):
    distList = []
    for i in range(encArr.shape[1] - 1):
        for j in range(i + 1, encArr.shape[1]):
            distList.append(np.linalg.norm(encArr[:, i] - encArr[:, j]))
    return sum(distList) / len(distList)

# Cosine similarity value between two vectors
def cos_sim_bt_vecs(vec1, vec2):
    cosSim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cosSim

# Get similarities of angles between an array of vectors and a single vector (assumes encoding_dim x samples)
def cos_sim_to_one_vec(encArr, encVec):
    simList = []
    for i in range(encArr.shape[1]):
        simList.append(cos_sim_bt_vecs(encArr[:, i], encVec))
    return simList

# Angle between two vectors using cosine similarity
def cos_ang_bt_vecs(vec1, vec2):
    cosSim = cos_sim_bt_vecs(vec1, vec2)
    return np.arccos(cosSim) * 180 / 3.14159

# Get list of angles between an array of vectors and a single vector (assumes encoding_dim x samples)
def cos_ang_to_one_vec(encArr, encVec):
    angList = []
    for i in range(encArr.shape[1]):
        angList.append(cos_ang_bt_vecs(encArr[:, i], encVec))
    return angList

# Get average angle between an array of vectors and a single vector (assumes encoding_dim x samples)
def avg_cos_ang_to_one_vec(encArr, encVec):
    angList = cos_ang_to_one_vec(encArr, encVec)
    return sum(angList) / len(angList)

# Get average angle between every pair of vectors in an array (assumes encoding_dim x samples)
def avg_cos_ang_bt_array(encArr):
    angList = []
    for i in range(encArr.shape[1] - 1):
        for j in range(i + 1, encArr.shape[1]):
            angList.append(cos_ang_bt_vecs(encArr[:, i], encArr[:, j]))
    return sum(angList) / len(angList)

# Get list of nonzero elements in vector
def one_vec_sparsity(encVec):
    nonZeroBool = np.abs(encVec) >= np.finfo(float).eps
    return nonZeroBool

# Get the number of nonzero elements for each vector in an array (assumes encoding_dim x samples)
def array_sparsity(encArr):
    sparsityList = []
    for i in range(encArr.shape[1]):
        nonZeroBool = np.abs(encArr[:, i]) >= np.finfo(float).eps
        sparsityList.append(sum(nonZeroBool))
    return sparsityList

# Get average number of nonzero elements for each vector in an array (assumes encoding_dim x samples)
def array_avg_sparsity(encArr):
    sparsityList = array_sparsity(encArr)
    return sum(sparsityList) / len(sparsityList)

# Get list of elements within 1 order of magnitude of underflow
def elem_near_zero(encVec):
    nearZeroBool = np.logical_and(np.abs(encVec) >= np.finfo(np.float32).eps,
                                  np.abs(encVec) <= 10 * np.finfo(np.float32).eps)
    return nearZeroBool

# Get the number of elements within 1 order of magnitude of underflow for each vector in an array (assumes encoding_dim x samples)
def array_near_zero(encArr):
    nearZeroList = []
    for i in range(encArr.shape[1]):
        nearZeroBool = np.logical_and(np.abs(encArr[:, i]) >= np.finfo(np.float32).eps,
                                      np.abs(encArr[:, i]) <= 10 * np.finfo(np.float32).eps)
        nearZeroList.append(sum(nearZeroBool))
    return nearZeroList

# Get average number of elements within 1 order of magnitude of underflow for each vector in an array (assumes encoding_dim x samples)
def array_avg_near_zero(encArr):
    nearZeroList = array_near_zero(encArr)
    return sum(nearZeroList) / len(nearZeroList)

# Get descending sorted eigval and eigvecs of the covariance matrix of array (assumes encoding_dim x samples)
def array_cov_eig(encArr):
    covArr = cross_cov(encArr, encArr)
    eigval, eigvec = np.linalg.eig(covArr)
    sortIdx = np.argsort(eigval)[::-1]
    sortedEigval = eigval[sortIdx]
    sortedEigvec = eigvec[:, sortIdx]
    return sortedEigval, sortedEigvec

# Get the cumulative sum values of a vector/list
def cumul_val(vals):
    cumulVal = np.cumsum(vals)
    return cumulVal

# Given a cumulative sum vector and its dx, get the area under the curve value
# Li et al. AUC uses 1/d (d=len(cumulVal)) factor but with dx of 1's - this function instead expects dx values sum to 1
def area_under_curve(cumulVal, dx):
    assert abs(sum(dx) - 1) < 0.0001
    auc = sum(cumulVal * dx) / cumulVal[-1]
    return auc

# Transform the array into covariance eigenspace, optionally reduce the eigenspace to a specified dimension (assumes encoding_dim x samples)
def array_to_reduced_eigspace(encArr, redDim=None):
    if redDim is None:
        redDim = encArr.shape[0]
    eigval, eigvec = array_cov_eig(encArr)
    reducedEigSpace = np.dot(eigvec[:, :redDim].T, encArr)
    return reducedEigSpace, eigval, eigvec

# Transform the array into covariance eigenspace, optionally reduce the eigenspace to a specified dimension, then reconstruct the original array (assumes encoding_dim x samples)
def eigspace_reduce_reconstruct(encArr, redDim=None):
    reducedEigSpace, eigval, eigvec = array_to_reduced_eigspace(encArr, redDim)
    reconstructedEncArr = np.dot(eigvec[:, :redDim], reducedEigSpace)
    return reconstructedEncArr, eigval, eigvec

##################
# FDA Projection #
##################

# Calculate transformation to the line that maximizes separation of projections for 2 distributions (assumes encoding_dim x samples)
def fisher_W(xArr1, xArr2):
    xMeanArr1 = np.mean(xArr1, axis=1, keepdims=True)
    xMeanArr2 = np.mean(xArr2, axis=1, keepdims=True)
    scatter1 = np.dot(xArr1 - xMeanArr1, np.transpose(xArr1 - xMeanArr1))
    scatter2 = np.dot(xArr2 - xMeanArr2, np.transpose(xArr2 - xMeanArr2))
    totalScatter = scatter1 + scatter2
    W = np.dot(np.linalg.inv(totalScatter), xMeanArr1 - xMeanArr2)
    W = W / np.linalg.norm(W)
    return W