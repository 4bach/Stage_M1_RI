import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir, "lib"))
sys.path.append(libpath)

print("Load libraries")
from itertools import chain
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, Subset

from data import load
from dataset import KeyWordSelectionDataset, sequence_collate_fn
from model import KeyWordSelectionModel_1a, memory_2c
from utils import all_but_one
import random
import numpy as np

seed = 666
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Constant definition
device = torch.device("cuda:0")
embedding_size = 300
hidden_size = 100
num_layers = 1
bidirectional = True

# Le probleme vient du count vectorizer qui vire certains mots
print("Load Dataset")
dataset = load(torch_dataset=True)["torch"]

def embedding_collate_decorator(collate_fn):
    def wrapper(batch):
        x, y, id_ = collate_fn(batch)
        return x, y
    return wrapper

collate_fn = embedding_collate_decorator(sequence_collate_fn)


indices = list(range(len(dataset)))
random.shuffle(indices)
for i, (trainindices, testindices) in enumerate(all_but_one(indices, k=10)):
    trainindices = chain(*trainindices)
    trainset = Subset(dataset, list(trainindices))
    testset = Subset(dataset, list(testindices))
    trainloader = DataLoader(trainset, 32, True, collate_fn=collate_fn)
    testloader = DataLoader(testset, 32, True, collate_fn=collate_fn)

    print("Build model")
    decoder_archi = {"input_size": embedding_size, "hidden_size": hidden_size, "num_layers": num_layers, "bidirectional":bidirectional, "dropout":0.2}

    model = memory_2c(KeyWordSelectionModel_1a)(decoder_archi, [2*hidden_size, 1])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    loss_function = nn.BCELoss()

    print("Train")
    best_model = 1e9
    delay = 0
    max_delay = 3
    for epoch in range(500):
        model.train()
        n, mean = 0, 0
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_function(pred, y.float())

            n += 1
            mean = ((n-1) * mean + loss.item()) / n
            print(f"\rFold {i}, Epoch {epoch}\tTrain : {mean}", end="")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()

        train_mean = mean
        n, mean = 0, 0
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_function(pred, y.float())

            n += 1
            mean = ((n-1) * mean + loss.item()) / n
            print(f"\rFold {i}, Epoch {epoch}\tTrain : {train_mean}\tTest : {mean}", end="")
        print()

        torch.save({"model_dict": model.state_dict()}, f"models_saves/fold{i}_1a2c_{epoch}_{mean}.torchsave")

        if mean < best_model:
            best_model = mean
            shutil.copyfile(f"models_saves/fold{i}_1a2c_{epoch}_{mean}.torchsave", f"models_saves/fold{i}_1a2c_best.torchsave")
            delay = 0
        else:
            delay += 1
            if delay > max_delay:
                break
