import torch
from torch.utils.data import Dataset

class En_Es_Dataset(Dataset):
    def __init__(self, enList, esList, tokenizer, vocabulary, seqLen, transform):
        self.enList = enList # List of english strings
        self.esList = esList # List of spanish strings
        self.tokenizer = tokenizer # Tokenizer object
        self.vocabulary = vocabulary # Vocabulary/Token dictionary
        self.seqLen = seqLen # Maximum sequence length
        self.transform = transform  # Useless atm

    def __len__(self):
        return len(self.enList)

    def __getitem__(self, index):

        # Transform the sample (somehow)
        if self.transform is not None:
            print('I do nothing')

        # Get the token index from the vocabulary dictionary for each token of the string
        enTokenList = [self.vocabulary[token] for token in self.tokenizer(self.enList[index])]
        esTokenList = [self.vocabulary[token] for token in self.tokenizer(self.esList[index])]

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
