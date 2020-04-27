from Model import spatioTemporalClassifier
from Dataloaders import getDataloader
import torch

#initialize the model, get the dataloader and train.

getDataloader(path='path', batch=16, workers=6)
model = spatioTemporalClassifier(classes=2)
