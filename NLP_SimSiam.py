#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#from model_checker import *

import math
from collections import Counter

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed

import simsiam.NLPSS_Builder
import simsiam.En_Es_Dataset

dataPath = 'D:\SpaEngTranslation\spa.txt' # Path to dataset
arch = 'rnn' # Encoder architecture
seed = None
nWorkers = 8
nEpochs = 20
startEpoch = 0
batchSize = 64
initLR = 0.05 # Initial LR before decay
momentum = 0.9
weightDecay = 0.0001
printFreq = 1859 # Number of batches before printing stats
checkpointPath = None # Path to resume from checkpoint
gpuID = 0 # GPU to use

seqLen = 32
embDim = 128
hidDim = 256
projDim = 256
predDim = 128
fixPredLR = False

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

enList, esList = get_string_lists(dataPath)
allList = enList.copy()
allList.extend(esList)

enTokenizer = get_tokenizer('spacy', language='en_core_web_sm')
#esTokenizer = get_tokenizer('spacy', language='es_core_news_sm')

def build_vocab(stringList, tokenizer, maxSize):
    counter = Counter()
    for stringVal in stringList:
        counter.update(tokenizer(stringVal))
    return vocab(dict(counter.most_common(maxSize)), specials=['<unk>', '<pad>'])

#enVocab = build_vocab(enList, enTokenizer, maxSize=10000)
#esVocab = build_vocab(esList, esTokenizer, maxSize=10000)
allVocab = build_vocab(allList, enTokenizer, maxSize=20000)
allVocab.set_default_index(allVocab['<unk>'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainDataset = simsiam.En_Es_Dataset.En_Es_Dataset(enList, esList, enTokenizer, allVocab, seqLen, transform=None)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=False)

model = simsiam.NLPSS_Builder.NLPSimSiam(len(allVocab), embDim=embDim, hidDim=hidDim, projDim=projDim, predDim=predDim)
model = model.to(device)

criterion = nn.CosineSimilarity(dim=1)
criterion = criterion.to(device)

initLR = initLR * batchSize / 256

if fixPredLR:
    optimParams = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                   {'params': model.module.predictor.parameters(), 'fix_lr': True}]
else:
    optimParams = model.parameters()

optimizer = torch.optim.SGD(optimParams, initLR, momentum=momentum, weight_decay=weightDecay)


def train(trainLoader, model, criterion, optimizer, epoch):

    model.train()
    lossAvg = 0

    for ii, phrases in enumerate(trainLoader):

        phrases[0] = phrases[0].transpose(1, 0).to(device)
        phrases[1] = phrases[1].transpose(1, 0).to(device)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=phrases[0], x2=phrases[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        lossAvg = (lossAvg * ii + loss.detach()) / (ii + 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ii == (len(trainLoader) - 1):
            print('Epoch {} | Batch {} / {} | Loss: {} | AvgLoss: {}'
                  .format(epoch, ii, len(trainLoader) - 1, loss.detach(), lossAvg))


def adjust_learning_rate(optimizer, initLR, epoch):
    """Decay the learning rate based on schedule"""
    cur_lr = initLR * 0.5 * (1. + math.cos(math.pi * epoch / nEpochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = initLR
        else:
            param_group['lr'] = cur_lr

for epoch in range(startEpoch, nEpochs):
    adjust_learning_rate(optimizer, initLR, epoch)
    train(trainLoader, model, criterion, optimizer, epoch)
