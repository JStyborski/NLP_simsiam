
import torch.nn as nn

class NLPSimSiam(nn.Module):

    def __init__(self, encArch='rnn', vocDim=20000, embDim=100, hidDim=256, projDim=256, predDim=128):
        # vocDim = vocabulary size
        # embDim = word embedding size
        # hidDim = RNN hidden dimension size
        # projDim = projector output size
        # predDim = predictor internal size

        super(NLPSimSiam, self).__init__()

        # Word embeddings
        self.embedder = nn.Embedding(vocDim, embDim)

        # The actual language model (RNN or transformer)
        if encArch == 'rnn':
            self.encoder = nn.RNN(embDim, hidDim)

        # 1-layer projector
        self.projector = nn.Sequential(nn.Linear(hidDim, projDim),
                                       nn.BatchNorm1d(projDim, affine=False))

        # 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(projDim, predDim, bias=False),
                                       nn.BatchNorm1d(predDim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(predDim, projDim))

    # Push both English/Spanish sentences through the network
    # Detach the projector output - this effectively applies stop-gradient to the Siamese side without predictor
    def forward(self, x1, x2):

        emb1 = self.embedder(x1)
        out1, hid1 = self.encoder(emb1)
        proj1 = self.projector(hid1.squeeze()) # Need to squeeze the first dimension out (corresponds to sequence dim)
        pred1 = self.predictor(proj1)

        emb2 = self.embedder(x2)
        out2, hid2 = self.encoder(emb2)
        proj2 = self.projector(hid2.squeeze())
        pred2 = self.predictor(proj2)
    
        return pred1, pred2, proj1.detach(), proj2.detach()
