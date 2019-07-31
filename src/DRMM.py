"""An implementation of DRMM Model."""
import typing
import torch
import numpy as np
import os 
import torch.utils.data as data


class Dataset(data.Dataset):

    def __init__(self,file):
        self.data = np.load(file,allow_pickle=True)

    def __getitem__(self,index):
        y = np.random.choice([0,1])
        return self.data[y][index], y

    def __len__(self):
        return len(self.data[0])


class Jujujul(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Jujujul, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # h_relu = self.linear1(x)#.clamp(min=0)
        # y_pred = self.linear2(h_relu)
        # return y_pred
        return torch.randn(x.size(0))


batch_size = 32
max_q = 8
nbins = 30
learning_rate = 10e-3

dataset = Dataset('/local/karmim/Stage_M1_RI/data/object_python/interaction/w2v_robust_all_concept/383LCH_interractions.npy')
#concatdataset


dataloader = data.DataLoader(concatdataset, batch_size=batch_size)
model = Jujujul(D_in=max_q, H = 15, D_out=1)
criterion = torch.nn.HingeEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    y = y.float()
    pred = model(x)
    loss = criterion(y, pred)
    loss.backward()
    optimizer.step()

    if i > 5:
        break
