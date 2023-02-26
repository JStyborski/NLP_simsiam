#from model_checker import *

import math
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
dataPath = 'D:\SpaEngTranslation\spa.txt' # Path to dataset
encArch = 'lstm' # Encoder architecture
seed = None # Seed number for RNG
nEpochs = 500
startEpoch = 0
batchSize = 256
initLR = 0.05 # Initial LR before decay
momentum = 0.9
weightDecay = 0.0001
checkpointPath = None # Path to resume from checkpoint - useless atm
fixPredLR = True # Fix the learning rate (no decay) of the predictor network
randAugment = True # Boolean to do random augmentation on sentences
augRNGThresh = 0.1 # Percent chance of a specific random augmentation occurring

seqLen = 20 # Permissible sentence length
vocDim = 10000 # Vocabulary size
embDim = 256 # Word embedding dimension
hidDim = 512 # RNN hidden dimension
projDim = 512 # Projector output dimension
predDim = 256 # Predictor internal dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if seed is not None:
    torch.seed(seed)
    cudnn.deterministic = True

#########################
# Preprocess Input Data #
#########################

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

# Create the sentence tokenizer (I just used the English one, it works well enough for tokenizing spanish too)
enTokenizer = get_tokenizer('spacy', language='en_core_web_sm')
#esTokenizer = get_tokenizer('spacy', language='es_core_news_sm')

# Count the number of unique tokens in the corpus, keep the most popular ones, then create a token dict
# Also add the <unk> and <pad> tokens to the dictonary
def build_vocab(stringList, tokenizer, maxSize):
    counter = Counter()
    for stringVal in stringList:
        counter.update(tokenizer(stringVal))
    return vocab(dict(counter.most_common(maxSize)), specials=['<unk>', '<pad>'])

# Get the token dictionary and set the default token as the <unk> token
#enVocab = build_vocab(enList, enTokenizer, maxSize=10000)
#esVocab = build_vocab(esList, esTokenizer, maxSize=10000)
allVocab = build_vocab(allList, enTokenizer, maxSize=vocDim)
allVocab.set_default_index(allVocab['<unk>'])

# Torch dataset/dataloader to gather samples, preprocess them, and load them as batches
# I think this can be improved with native torchtext functions - I set it up like a custom image dataset to make batches
trainDataset = simsiam.En_Es_Dataset.En_Es_Dataset(enList, esList, enTokenizer, allVocab, seqLen, randAugment, augRNGThresh)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=False)

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

# Adjust the intial learning rate based on batch size
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

        # Print status
        if ii == (len(trainLoader) - 1):
            print('Epoch {:04d} | Batch {:04d} / {:04d} | 1 Batch Loss: {:6.3f} | Epoch Avg Loss: {:6.3f}'
                  .format(epoch, ii, len(trainLoader) - 1, loss.detach(), lossAvg))

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

# Do the actual training
for epoch in range(startEpoch, nEpochs):
    adjust_learning_rate(optimizer, initLR, epoch)
    train(trainLoader, model, criterion, optimizer, epoch)

    if (epoch + 1) % 50 == 0:
        save_checkpoint({'epoch': epoch,
                         'params': {'encArch': encArch, 'nEpochs': nEpochs, 'batchSize': batchSize, 'initLR': initLR,
                                    'momentum': momentum, 'weightDecay': weightDecay, 'fixPredLR': fixPredLR,
                                    'randAugment': randAugment, 'augRNGThresh': augRNGThresh, 'seq': seqLen,
                                    'voc': vocDim, 'emb': embDim, 'hid': hidDim, 'proj': projDim, 'pred': predDim},
                         'tokenizer': enTokenizer,
                         'vocabulary': allVocab,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        fileName='checkpoints\checkpoint{:04d}.pth.tar'.format(epoch))
