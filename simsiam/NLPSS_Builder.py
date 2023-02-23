
import torch.nn as nn

class NLPSimSiam(nn.Module):

    def __init__(self, vocDim, embDim=100, hidDim=256, projDim=256, predDim=128):

        super(NLPSimSiam, self).__init__()

        self.embedder = nn.Embedding(vocDim, embDim)

        self.rnn = nn.RNN(embDim, hidDim)

        # Build a 1-layer projector
        self.projector = nn.Sequential(nn.Linear(hidDim, projDim),
                                       nn.BatchNorm1d(projDim, affine=False))

        # Build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(projDim, predDim, bias=False),
                                       nn.BatchNorm1d(predDim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(predDim, projDim))

    def forward(self, x1, x2):

        emb1 = self.embedder(x1)
        out1, hid1 = self.rnn(emb1)
        proj1 = self.projector(hid1.squeeze())
        pred1 = self.predictor(proj1)

        emb2 = self.embedder(x2)
        out2, hid2 = self.rnn(emb2)
        proj2 = self.projector(hid2.squeeze())
        pred2 = self.predictor(proj2)
    
        return pred1, pred2, proj1.detach(), proj2.detach()
