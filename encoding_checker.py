
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
dataPath = r'C:\Users\jeremy\Python_Projects\NLP_simsiam\spa.txt' # Path to dataset
seed = None # Seed number for RNG
batchSize = 256
checkpointPath = 'checkpoints\checkpoint0499_LSTM_5pctAug_Punct_WordToken_0p25LR.pth.tar' # Path to resume from checkpoint - useless atm

nSamples = 20000 # Number of samples to downsample testing samples - set as None to use all

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if seed is not None:
    torch.seed(seed)
    cudnn.deterministic = True

###################
# Load Checkpoint #
###################

# Hack - have to define character tokenizer in case it's referenced in the checkpoint
def charTokenizer(string):
    return [char for char in string]

checkpoint = torch.load(checkpointPath, map_location='cpu')

encArch = checkpoint['params']['encArch']
seqLen = checkpoint['params']['seqLen']
vocDim = checkpoint['params']['vocDim']
embDim = checkpoint['params']['embDim']
hidDim = checkpoint['params']['hidDim']
projDim = checkpoint['params']['projDim']
predDim = checkpoint['params']['predDim']
noPunct = checkpoint['params']['noPunct']
randAugment = checkpoint['params']['randAugment']
stateDict = checkpoint['state_dict']
tokenizer = checkpoint['tokenizer']
vocabulary = checkpoint['vocabulary']

######################
# Get Data Encodings #
######################

# From the text file, gather the corresponding lists of English/Spanish sentences
def get_string_lists(filePath, noPunct=False):
    with open(filePath, 'r', encoding='utf-8') as f:
        allLines = f.readlines()

    enList = []
    esList = []
    for line in allLines:
        lineVals = line.rstrip().lower().split('\t')
        if noPunct:
            lineVals[0] = ''.join(char for char in lineVals[0] if char.isalnum() or char == ' ')
            lineVals[1] = ''.join(char for char in lineVals[1] if char.isalnum() or char == ' ')
        enList.append(lineVals[0])
        esList.append(lineVals[1])

    return enList, esList

# Get the string lists and then make one large list of both English and Spanish
enList, esList = get_string_lists(dataPath, noPunct)

# Downsample data lists to a smaller set of samples (for quicker compute)
if nSamples is not None:
    nRandIdx = np.random.choice(list(range(len(enList))), size=nSamples, replace=False)
    enList = [enList[idx] for idx in nRandIdx]
    esList = [esList[idx] for idx in nRandIdx]

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
tokArr = np.zeros((seqLen, 1))
encArr = np.zeros((hidDim, 1))

model.eval()
with torch.no_grad():
    for i, phrases in enumerate(trainLoader):

        tokArr = np.concatenate((tokArr, phrases[0].numpy().transpose()), axis=1)
        tokArr = np.concatenate((tokArr, phrases[1].numpy().transpose()), axis=1)
        startIdx = i * batchSize
        endIdx = startIdx + phrases[0].size()[0]
        stringList.extend(enList[startIdx:endIdx])
        stringList.extend(esList[startIdx:endIdx])
        enTokens = phrases[0].transpose(1, 0).to(device)
        esTokens = phrases[1].transpose(1, 0).to(device)

        enEmb = model.embedder(enTokens)
        esEmb = model.embedder(esTokens)

        if encArch == 'rnn':
            enOut, enHid = model.encoder(enEmb)
            esOut, esHid = model.encoder(esEmb)
            enHid = enHid.squeeze()
            esHid = esHid.squeeze()
        elif encArch == 'lstm':
            enOut, enHid = model.encoder(enEmb)
            esOut, esHid = model.encoder(esEmb)
            enHid = enHid[0].squeeze()
            esHid = esHid[0].squeeze()
        elif encArch == 'bilstm':
            enOut, enHid = model.encoder(enEmb)
            esOut, esHid = model.encoder(esEmb)
            enHid = torch.concat([enHid[0][0, :, :], enHid[0][1, :, :]], dim=1).squeeze()
            esHid = torch.concat([esHid[0][0, :, :], esHid[0][1, :, :]], dim=1).squeeze()
        elif encArch == 'cnn':
            enEmb = torch.permute(enEmb, (1, 2, 0))
            esEmb = torch.permute(esEmb, (1, 2, 0))
            enHid = torch.mean(model.encoder(enEmb), dim=-1, keepdim=False)
            esHid = torch.mean(model.encoder(esEmb), dim=-1, keepdim=False)
        elif encArch == 'ffn':
            enHid = model.encoder(torch.mean(enEmb, dim=0, keepdim=False))
            esHid = model.encoder(torch.mean(esEmb, dim=0, keepdim=False))

        encArr = np.concatenate((encArr, enHid.cpu().numpy().transpose()), axis=1)
        encArr = np.concatenate((encArr, esHid.cpu().numpy().transpose()), axis=1)

tokArr = tokArr[:, 1:]
encArr = encArr[:, 1:]

#####################
# Analyze Encodings #
#####################

# Get the encArr idxs sorted by closest to the encoding at idx
def get_closest_enc_idxs(encArr, idx):
    cosSimList = cos_sim_to_one_vec(encArr, encArr[:, idx])
    sortIdxs = np.argsort(cosSimList)[::-1]
    return sortIdxs

# Using encArr indices sorted by closest, get the topk of them
def get_closest_vecs(idx, encArr, topk):
    sortIdxs = get_closest_enc_idxs(encArr, idx)
    topkIdxs = sortIdxs[0:topk]

    return topkIdxs

# Using encArr indices sorted by closest, get the topk of them that are in the opposite language
def get_closest_one_lang_vecs(idx, oppList, encArr, topk):
    sortIdxs = get_closest_enc_idxs(encArr, idx)

    topkIdxs = []
    for sortIdx in sortIdxs:
        if stringList[sortIdx] in oppList:
            topkIdxs.append(sortIdx)
        if len(topkIdxs) >= topk:
            break

    return topkIdxs

# Print the queried string and the topk closest strings - may be any lang or specified language
def print_topk_strings(idx, langList, encArr, topk):
    if langList is None:
        topkIdxs = get_closest_vecs(idx, encArr, topk)
    else:
        topkIdxs = get_closest_one_lang_vecs(idx, langList, encArr, topk)

    topkStrings = [stringList[idx] for idx in topkIdxs]

    print('\nQuery: {}'.format(stringList[idx]))
    print('Closest sentences:')
    for stringItem in topkStrings:
        print(stringItem)

# Calculate accuracy of the true translation being in the topk nearest translations
def get_top_n_acc(encArr, nQueries, topk):
    nRandIdx = np.random.choice(list(range(encArr.shape[1])), size=nQueries, replace=False)

    inTopkCount = 0
    for idx in nRandIdx:

        queryString = stringList[idx]
        if queryString in enList:
            stringIdx = enList.index(queryString)
            oppList = esList
        else:
            stringIdx = esList.index(queryString)
            oppList = enList

        topkIdxs = get_closest_one_lang_vecs(idx, oppList, encArr, topk)
        topkStrings = [stringList[idx] for idx in topkIdxs]

        if oppList[stringIdx] in topkStrings:
            inTopkCount += 1

    return inTopkCount / nQueries

topk = 5
print_topk_strings(1000, None, encArr, topk)
print_topk_strings(1000, enList, encArr, topk)
print_topk_strings(1000, esList, encArr, topk)
print_topk_strings(2200, None, encArr, topk)
print_topk_strings(2200, enList, encArr, topk)
print_topk_strings(2200, esList, encArr, topk)
print_topk_strings(10000, None, encArr, topk)
print_topk_strings(10000, enList, encArr, topk)
print_topk_strings(10000, esList, encArr, topk)

topn = 10
topNAcc = get_top_n_acc(encArr, 500, topn)
print('\nTop-{} Acc: {}'.format(topn, topNAcc))