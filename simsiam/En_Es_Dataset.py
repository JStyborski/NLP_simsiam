import torch
from torch.utils.data import Dataset

class En_Es_Dataset(Dataset):
    def __init__(self, enList, esList, tokenizer, vocabulary, seqLen, transform):
        self.transform = transform
        self.enList = enList
        self.esList = esList
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.seqLen = seqLen

    def __len__(self):
        return len(self.enList)

    def __getitem__(self, index):

        if self.transform is not None:
            print('I do nothing')

        enTokenList = [self.vocabulary[token] for token in self.tokenizer(self.enList[index])]
        esTokenList = [self.vocabulary[token] for token in self.tokenizer(self.esList[index])]

        if len(enTokenList) < self.seqLen:
            enTokenList = [self.vocabulary['<pad>']] * (self.seqLen - len(enTokenList)) + enTokenList
        else:
            enTokenList = enTokenList[:self.seqLen]
        if len(esTokenList) < self.seqLen:
            esTokenList = [self.vocabulary['<pad>']] * (self.seqLen - len(esTokenList)) + esTokenList
        else:
            esTokenList = esTokenList[:self.seqLen]

        enTens = torch.tensor(enTokenList)
        esTens = torch.tensor(esTokenList)

        return enTens, esTens
