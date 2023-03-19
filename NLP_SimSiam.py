#from model_checker import *

import math
import random
from collections import Counter

import torch.nn as nn
import torch.utils.data
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
import torch.backends.cudnn as cudnn

import simsiam.En_Es_Dataset
import simsiam.NLPSS_Builder

###############
# User Inputs #
###############

# Dataset from http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
dataPath = r'C:\Users\jeremy\Python_Projects\NLP_simsiam\spa.txt' # Path to dataset
encArch = 'bilstm' # Encoder architecture
seed = None # Seed number for RNG
nEpochs = 500
startEpoch = 0
nSubsets = 1 # Number of subsets do divide dataset into for single-pass training - if = 1, regular multipass training
batchSize = 256
initLR = 0.05 # Initial LR before decay
momentum = 0.9
weightDecay = 0.0001
checkpointPath = None # Path to resume from checkpoint - useless atm
fixPredLR = True # Fix the learning rate (no decay) of the predictor network
charVocab = False # Use character level vocabulary, not word level
noPunct = False # Strip punctuation from sentences
randAugment = False # Boolean to do random augmentation on sentences
augRNGThresh = 0.1 # Percent chance of a specific random augmentation occurring

seqLen = 20 # Permissible sentence length
vocDim = 10000 # Vocabulary size
embDim = 128 # Word embedding dimension
hidDim = 256 # Hidden dimension
projDim = 256 # Projector output dimension
predDim = 128 # Predictor internal dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if seed is not None:
    torch.seed(seed)
    cudnn.deterministic = True

#########################
# Preprocess Input Data #
#########################

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
allList = enList.copy()
allList.extend(esList)

# Create the sentence tokenizer (I just used the English one, it works well enough for tokenizing spanish too)
if charVocab:
    def charTokenizer(string):
        return [char for char in string]
    tokenizer = charTokenizer
else:
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    #esTokenizer = get_tokenizer('spacy', language='es_core_news_sm')

# Count the number of unique tokens in the corpus, keep the most popular ones, then create a token dict
# Also add the <unk> and <pad> tokens to the dictionary
def build_vocab(stringList, tokenizer, maxSize):
    counter = Counter()
    for stringVal in stringList:
        counter.update(tokenizer(stringVal))
    return vocab(dict(counter.most_common(maxSize)), specials=['<unk>', '<pad>'])

# Get the token dictionary and set the default token as the <unk> token
#enVocab = build_vocab(enList, enTokenizer, maxSize=10000)
#esVocab = build_vocab(esList, esTokenizer, maxSize=10000)
allVocab = build_vocab(allList, tokenizer, maxSize=vocDim)
allVocab.set_default_index(allVocab['<unk>'])

###############
# Build Model #
###############

# Build the NLP_SimSiam model
model = simsiam.NLPSS_Builder.NLPSimSiam(encArch=encArch, vocDim=len(allVocab), embDim=embDim,
                                         hidDim=hidDim, projDim=projDim, predDim=predDim)
model = model.to(device)

# Same as SimSiam
criterion = nn.CosineSimilarity(dim=1)
criterion = criterion.to(device)

# Adjust the initial learning rate based on batch size
initLR = initLR * batchSize / 256

# Fix the learning rate of the predictor
if fixPredLR:
    optimParams = [{'params': model.embedder.parameters(), 'fix_lr': False},
                   {'params': model.encoder.parameters(), 'fix_lr': False},
                   {'params': model.projector.parameters(), 'fix_lr': False},
                   {'params': model.predictor.parameters(), 'fix_lr': True}]
else:
    optimParams = model.parameters()

# Set the model optimizer
optimizer = torch.optim.SGD(optimParams, initLR, momentum=momentum, weight_decay=weightDecay)

###############
# Train Model #
###############

# Function to carry out 1 epoch of training
def train(trainLoader, model, criterion, optimizer, epoch):

    model.train()
    lossAvg = 0

    for ii, phrases in enumerate(trainLoader):

        # Get batched inputs in correct format (seqLen, batch size)
        phrases[0] = phrases[0].transpose(1, 0).to(device)
        phrases[1] = phrases[1].transpose(1, 0).to(device)

        # Compute output and loss
        p1, p2, z1, z2 = model(x1=phrases[0], x2=phrases[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        lossAvg = (lossAvg * ii + loss.detach()) / (ii + 1)

        # Gradient and backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.detach(), lossAvg

# Decay the learning rate based on schedule
def adjust_learning_rate(optimizer, initLR, epoch):
    cur_lr = initLR * 0.5 * (1. + math.cos(math.pi * epoch / nEpochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = initLR
        else:
            param_group['lr'] = cur_lr

# Save the model data
def save_checkpoint(state, fileName='checkpoint.pth.tar'):
    torch.save(state, fileName)

# Torch dataset/dataloader to gather samples, preprocess them, and load them as batches
# I think this can be improved with native torchtext functions - I set it up like a custom image dataset to make batches
trainDataset = simsiam.En_Es_Dataset.En_Es_Dataset(enList, esList, tokenizer, allVocab,
                                                   seqLen, randAugment, augRNGThresh)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=False)

# Sets up subsets for single-pass training on training data subsets
# Generate a list of randomized indices of the length of the dataset and denote which ones correspond to each subset
if nSubsets > 1:
    randIdxList = list(range(len(trainDataset)))
    random.shuffle(randIdxList)
    subsetSize = math.floor(len(randIdxList) / nSubsets)
    subsetIdx = {}
    for i in range(nSubsets):
        if i == nSubsets - 1:
            subsetIdx[i] = [i * subsetSize, len(randIdxList)]
        else:
            subsetIdx[i] = [i * subsetSize, (i + 1) * subsetSize]

# Do the actual training
for epoch in range(startEpoch, nEpochs):
    adjust_learning_rate(optimizer, initLR, epoch)

    # If doing single-pass training and ready for a new training subset, then create the new subset and dataloader
    # Only updates subset/dataloader every nEpochs/nSubsets epochs
    if nSubsets > 1 and epoch % (nEpochs / nSubsets) == 0:
        subsetNum = math.floor(epoch / (nEpochs / nSubsets))
        trainSubset = torch.utils.data.Subset(trainDataset,
                                              randIdxList[subsetIdx[subsetNum][0]:subsetIdx[subsetNum][1]])
        trainLoader = torch.utils.data.DataLoader(trainSubset, batch_size=batchSize, shuffle=True, drop_last=False)

    # Train the model using the current trainloader
    # If using multiple smaller subsets, train nSubsets times on the 1/nSubsets sized dataset
    for _ in range(nSubsets):
        loss, lossAvg = train(trainLoader, model, criterion, optimizer, epoch)

    # Print status
    outString = 'Epoch {:04d} | 1 Batch Loss: {:6.3f} | Epoch Avg Loss: {:6.3f}'.format(epoch, loss.detach(), lossAvg)
    print(outString)
    with open('checkpoints\out.txt', 'a') as f:
        f.write(outString + '\n')

    # Save checkpoint
    if (epoch + 1) % 100 == 0:
        save_checkpoint({'epoch': epoch,
                         'params': {'encArch': encArch, 'nEpochs': nEpochs, 'nSubsets': nSubsets,
                                    'batchSize': batchSize, 'initLR': initLR, 'momentum': momentum,
                                    'weightDecay': weightDecay, 'fixPredLR': fixPredLR, 'noPunct': noPunct,
                                    'randAugment': randAugment, 'augRNGThresh': augRNGThresh, 'seqLen': seqLen,
                                    'vocDim': vocDim, 'embDim': embDim, 'hidDim': hidDim, 'projDim': projDim,
                                    'predDim': predDim},
                         'tokenizer': tokenizer,
                         'vocabulary': allVocab,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        fileName='checkpoints\checkpoint{:04d}.pth.tar'.format(epoch))
