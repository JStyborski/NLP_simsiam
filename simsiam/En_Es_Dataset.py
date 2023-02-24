import torch
from torch.utils.data import Dataset

import numpy as np

class En_Es_Dataset(Dataset):
    def __init__(self, enList, esList, tokenizer, vocabulary, seqLen, randAugment, augRNGThresh):
        self.enList = enList # List of english strings
        self.esList = esList # List of spanish strings
        self.tokenizer = tokenizer # Tokenizer object
        self.vocabulary = vocabulary # Vocabulary/Token dictionary
        self.seqLen = seqLen # Maximum sequence length
        self.randAugment = randAugment # Boolean to do random augmentation
        self.augRNGThresh = augRNGThresh # % chance of a specific augmentation of occuring

    def __len__(self):
        return len(self.enList)

    def __getitem__(self, index):

        # Get the token index from the vocabulary dictionary for each token of the string
        enTokenList = [self.vocabulary[token] for token in self.tokenizer(self.enList[index])]
        esTokenList = [self.vocabulary[token] for token in self.tokenizer(self.esList[index])]

        if self.randAugment:
            randRolls = np.random.random(6)
            # Remove a random token
            if randRolls[0] <= self.augRNGThresh:
                enTokenList.pop(np.random.randint(len(enTokenList)))
            if randRolls[1] <= self.augRNGThresh:
                esTokenList.pop(np.random.randint(len(esTokenList)))
            # Add a random token at a random position
            if randRolls[2] <= self.augRNGThresh:
                enTokenList.insert(np.random.randint(len(enTokenList) + 1), np.random.randint(len(self.vocabulary)))
            if randRolls[3] <= self.augRNGThresh:
                esTokenList.insert(np.random.randint(len(esTokenList) + 1), np.random.randint(len(self.vocabulary)))
            # Replace a token at a random position with a random token
            if randRolls[4] <= self.augRNGThresh:
                enTokenList[np.random.randint(len(enTokenList))] = np.random.randint(len(self.vocabulary))
            if randRolls[5] <= self.augRNGThresh:
                esTokenList[np.random.randint(len(esTokenList))] = np.random.randint(len(self.vocabulary))

        # If under sequence length, pad beginning until full. If over sequence length, cut off
        if len(enTokenList) < self.seqLen:
            enTokenList = [self.vocabulary['<pad>']] * (self.seqLen - len(enTokenList)) + enTokenList
        else:
            enTokenList = enTokenList[:self.seqLen]
        if len(esTokenList) < self.seqLen:
            esTokenList = [self.vocabulary['<pad>']] * (self.seqLen - len(esTokenList)) + esTokenList
        else:
            esTokenList = esTokenList[:self.seqLen]

        # Convert to torch tensors
        enTens = torch.tensor(enTokenList)
        esTens = torch.tensor(esTokenList)

        return enTens, esTens
