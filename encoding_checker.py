
import numpy as np

import simsiam.En_Es_Dataset
import simsiam.NLPSS_Builder
from Encoding_Checker_Utils import *

import torch
import torch.backends.cudnn as cudnn

###############
# User Inputs #
###############

# Dataset from http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
dataPath = 'D:\SpaEngTranslation\spa.txt' # Path to dataset
seed = None # Seed number for RNG
batchSize = 256
checkpointPath = 'checkpoints\checkpoint0999_5pctAug.pth.tar' # Path to resume from checkpoint - useless atm

nSamples = None # Number of samples to downsample encodings - set as None to use all
topk = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if seed is not None:
    torch.seed(seed)
    cudnn.deterministic = True

###################
# Load Checkpoint #
###################

checkpoint = torch.load(checkpointPath, map_location='cpu')

encArch = checkpoint['params']['encArch']
seqLen = checkpoint['params']['seqLen']
vocDim = checkpoint['params']['vocDim']
embDim = checkpoint['params']['embDim']
hidDim = checkpoint['params']['hidDim']
projDim = checkpoint['params']['projDim']
predDim = checkpoint['params']['predDim']
randAugment = checkpoint['params']['randAugment']
stateDict = checkpoint['state_dict']
tokenizer = checkpoint['tokenizer']
vocabulary = checkpoint['vocabulary']

######################
# Get Data Encodings #
######################

# From the text file, gather the corresponding lists of English/Spanish sentences
def get_string_lists(filePath):
    with open(filePath, 'r', encoding='utf-8') as f:
        allLines = f.readlines()

    enList = []
    esList = []
    for line in allLines:
        lineVals = line.rstrip().lower().split('\t')
        enList.append(lineVals[0])
        esList.append(lineVals[1])

    return enList, esList

# Get the string lists and then make one large list of both English and Spanish
enList, esList = get_string_lists(dataPath)
allList = enList.copy()
allList.extend(esList)

# Torch dataset/dataloader to gather samples, preprocess them, and load them as batches
# I think this can be improved with native torchtext functions - I set it up like a custom image dataset to make batches
trainDataset = simsiam.En_Es_Dataset.En_Es_Dataset(enList, esList, tokenizer, vocabulary, seqLen, randAugment)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=False, drop_last=False)

model = simsiam.NLPSS_Builder.NLPSimSiam(encArch=encArch, vocDim=len(vocabulary), embDim=embDim,
                                         hidDim=hidDim, projDim=projDim, predDim=predDim)
model = model.to(device)
model.load_state_dict(stateDict)

stringList = []
tokArr = np.zeros((1, seqLen))
encArr = np.zeros((1, hidDim))

model.eval()
with torch.no_grad():
    for i, phrases in enumerate(trainLoader):
        if i % 100 == 0:
            print('{} lines processsed'.format(i * batchSize))

        tokArr = np.concatenate((tokArr, phrases[0].numpy()), axis=0)
        tokArr = np.concatenate((tokArr, phrases[1].numpy()), axis=0)
        startIdx = i * batchSize
        endIdx = startIdx + phrases[0].size()[0]
        stringList.extend(enList[startIdx:endIdx])
        stringList.extend(esList[startIdx:endIdx])
        enTokens = phrases[0].transpose(1, 0).to(device)
        esTokens = phrases[1].transpose(1, 0).to(device)

        enEmb = model.embedder(enTokens)
        esEmb = model.embedder(esTokens)
        enOut, enHid = model.encoder(enEmb)
        esOut, esHid = model.encoder(esEmb)

        if encArch == 'lstm':
            enHid = enHid[0]
            esHid = esHid[0]
        encArr = np.concatenate((encArr, enHid.squeeze().cpu().numpy()), axis=0)
        encArr = np.concatenate((encArr, esHid.squeeze().cpu().numpy()), axis=0)

tokArr = tokArr[1:, :]
encArr = encArr[1:, :]

#####################
# Analyze Encodings #
#####################

#avgEucDist = avg_euc_dist_between_array(encArr)
#avgCosAng = avg_cos_ang_between_array(encArr)

#print(avgEucDist)
#print(avgCosAng)

if nSamples is not None:
    randIdx = np.random.randint(0, len(stringList), nSamples)
    stringList = stringList[randIdx]
    tokArr = tokArr[randIdx, :]
    encArr = encArr[randIdx, :]

def get_closest_vecs(idx, encArr, topk):
    cosAngsList = cos_ang_to_one_vec(encArr, encArr[idx, :])
    sortIdxs = np.argsort(cosAngsList)
    print()

    print('\nQuery: {}'.format(stringList[idx]))

    print('Closest sentences:')
    for idx in sortIdxs[0:topk]:
        print(stringList[idx])

def get_closest_one_lang_vecs(idx, oppList, encArr, topk):
    cosAngsList = cos_ang_to_one_vec(encArr, encArr[idx, :])
    sortIdxs = np.argsort(cosAngsList)
    print()

    print('\nQuery: {}'.format(stringList[idx]))

    print('Closest sentences:')
    count = 0
    for sortIdx in sortIdxs:
        if stringList[sortIdx] in oppList:
            print(stringList[sortIdx])
            count += 1
        if count >= topk:
            break

get_closest_vecs(1000, encArr, topk)
get_closest_one_lang_vecs(1000, enList, encArr, topk)
get_closest_one_lang_vecs(1000, esList, encArr, topk)
get_closest_vecs(10000, encArr, topk)
get_closest_one_lang_vecs(10000, enList, encArr, topk)
get_closest_one_lang_vecs(10000, esList, encArr, topk)
get_closest_vecs(20000, encArr, topk)
get_closest_one_lang_vecs(20000, enList, encArr, topk)
get_closest_one_lang_vecs(20000, esList, encArr, topk)
get_closest_vecs(100000, encArr, topk)
get_closest_one_lang_vecs(100000, enList, encArr, topk)
get_closest_one_lang_vecs(100000, esList, encArr, topk)
