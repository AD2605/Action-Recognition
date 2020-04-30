from Model import spatioTemporalClassifier
from Dataloaders import getDataloader, ufctraintest

#if frame counts are different then batch size can only be 1
data = getDataloader('path', batch=2, workers=6, frames=102)
model = spatioTemporalClassifier(classes=2)
model.train_model(model=model, dataloader=data, epochs=2)
