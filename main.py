from Model import spatioTemporalClassifier
from Dataloaders import getDataloader

data = getDataloader('path', batch='batch', workers=6)
model = spatioTemporalClassifier(classes=2)
model.train_model(model=model, dataloader=data, epochs=2)
