"""An implementation of DRMM Model."""
import typing
import torch
import numpy as np
import os 
import torch.utils.data as data


class Dataset(data.Dataset):

    def __init__(self,file):
        self.data = np.load(file,allow_pickle=True)
        self.label = True
    def __getitem__(self,index):
        y = int(self.label)
        self.label = not self.label
        #y = np.random.choice([0,1])
        return self.data[y][index], y

    def __len__(self):
        return len(self.data[0])


class Jujujul(torch.nn.Module):
    def __init__(self, D_in=30, H1=5, H2=8, D_out=1):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Jujujul, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)
        self.linear3 = torch.nn.Linear(H2, D_out)
        self.linear4 = torch.nn.Linear(D_out, D_out)
        self.activ = torch.nn.Softmax(-1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y = self.linear1(x)
        y = self.activ(y)
        
        y = self.linear2(y).squeeze()
        y = self.activ(y)
        
        y = self.linear3(y)
        y = self.activ(y)
        
        y = self.linear4(y).squeeze()
        y = self.activ(y)

        return y


batch_size = 32
max_q = 8
nbins = 30
learning_rate = 10e-3
n_epoch=1000
hist_type = 'LCH'
directory = '/local/karmim/Stage_M1_RI/data/object_python/interaction/w2v_robust_all_concept/'
files = [os.path.join(directory, f) for f in os.listdir(directory) if hist_type in f]
datasets = [Dataset(inter_file) for inter_file in files][:1] 
print('remove')
concatdataset = torch.utils.data.ConcatDataset(datasets)


dataloader = data.DataLoader(concatdataset, batch_size=batch_size)
model = Jujujul(D_in=30, H1=5, H2=8, D_out=1).double()
criterion = torch.nn.HingeEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for _ in range(n_epoch):
    for i, (x, y) in enumerate(dataloader):
        model.train()
        y = y.float()
        x = x.double()
        optimizer.zero_grad()
        pred = model(x)
        print(pred)
        loss = criterion(pred, y)
        print('loss', loss)
        loss.backward()
        optimizer.step()
