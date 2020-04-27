import cv2
import numpy
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import os

class customVideoDataset(Dataset):
    def __init__(self, path):
        self.videos = []
        self.labels = []
        folder = Path(path)
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.videos.append(os.path.join(folder, label, fname))
                self.labels.append(label)

        self.label2index = {label:index for index, label in enumerate(sorted(set(self.labels)))}
        self.label_array = numpy.array([self.label2index[label] for label in self.labels], dtype=int)

    def __getitem__(self, idx):
        video = cv2.VideoCapture(self.videos[idx])
        depth = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        heigth = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stacked_frames = numpy.empty(shape=(depth, heigth, width, 3), dtype=numpy.dtype('float16')) #as frame would have shape h,w,channels
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count = frame_count+1
            if not ret:
                break
            frame = cv2.resize(frame, (128 ,128))
            stacked_frames[frame_count] = frame
        video.release()
        stacked_frames = stacked_frames.transpose((3, 0, 1, 2))

        return stacked_frames, self.label_array[idx]

    def __len__(self):
        length = len(self.videos)
        return length

def getDataloader(path, batch, workers):
    dataset = customVideoDataset(path=path)
    dataloader = DataLoader(dataset,batch_size=batch, num_workers=workers, shuffle=True)
    return dataloader