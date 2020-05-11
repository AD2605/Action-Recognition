import torch.nn as nn
import torch

class k1_convolution(nn.Module):
    def __init__(self, input_planes, output_planes):
        super(k1_convolution, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv3d(input_planes,
                                     output_planes,
                                     kernel_size=1,
                                     stride=1,
                                     padding = (1,1,1)
                                     ))
        self.layers.append(nn.BatchNorm3d(output_planes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return nn.functional.relu6(x)

class inverted_residuals(nn.Module):
    def __init__(self, input_planes, output_planes, stride, expantionRatio):
        super(inverted_residuals, self).__init__()
        self.intermediate_dims = input_planes*expantionRatio

        if expantionRatio ==1:
            self.convolultion = nn.Sequential(
                nn.Conv3d(self.intermediate_dims,
                          self.intermediate_dims,
                          kernel_size=3,
                          stride=stride,
                          groups=self.intermediate_dims,
                          padding=(1,1,1)),
                nn.BatchNorm3d(self.intermediate_dims),
                nn.ReLU6(inplace=True),
                nn.Conv3d(input_planes, output_planes, 1, 1, 0),
                nn.BatchNorm3d(output_planes)
            )

        else:
            self.convolultion = nn.Sequential(
                nn.Conv3d(input_planes,
                          self.intermediate_dims,
                          1, 1, 0
                          ),
                nn.BatchNorm3d(self.intermediate_dims),
                nn.ReLU6(inplace=True),
                nn.Conv3d(self.intermediate_dims,
                          self.intermediate_dims,
                          kernel_size=3,
                          stride=stride,
                          groups=self.intermediate_dims,
                          padding=(1,1,1)),
                nn.BatchNorm3d(self.intermediate_dims),
            )
    def forward(self, x):
        return  x + self.convolultion(x)

class MobilenetV2(nn.Module):
    def __intit(self, classes, expantionRatio, inputFrames):
        super(MobilenetV2, self).__intit()
        self.input_planes = inputFrames
        self.end_channel = 1280*expantionRatio
        self.expanded_input_planes = self.input_planes*expantionRatio
        self.expanded_end_channels = self.end_channel*expantionRatio

        self.first_conv = nn.Sequential(
            nn.Conv3d(in_channels=3,
                      out_channels=inputFrames*expantionRatio,
                      kernel_size=3,
                      padding=(1, 1, 1),
                      stride=(1, 1, 1)),
            nn.BatchNorm3d(inputFrames),
            nn.ReLU6(inplace=True)
        )
        self.residuals = nn.ModuleList()
        num_channels = [[16, (1, 1, 1), 1],
                        [24, (2, 2, 2), 2],
                        [32, (2, 2, 2), 3],
                        [64, (2, 2, 2), 4],
                        [96, (1, 1, 1), 3],
                        [160, (2, 2, 2), 3],
                        [320, (1, 1, 1), 1]
                        ]

        for output, stride, channel in num_channels:
            if output == 16:
                expantion_Ratio = 1
            else:
                expantion_Ratio = 6
            for i in range(channel+1):
                stride = (1, 1, 1) if not i==0 else stride= stride
                self.residuals.append(inverted_residuals(
                    input_planes= self.input_planes*expantion_Ratio,
                    output_planes= expantionRatio*output,
                    stride=stride,
                    expantionRatio=expantion_Ratio
                ))
                self.input_planes = expantion_Ratio*output

        self.residuals.append(k1_convolution(self.input_planes,
                                             self.end_channel
                                             ))
        self.linear = nn.Linear(in_features=self.end_channel, out_features=classes)

    def train_model(self, model, dataloader, epochs):
        model.train()
        model.cuda()
        optmizer = torch.optim.Adam(model.parameters(), lr =0.001)
        criterion = nn.CrossEntropyLoss().cuda()

        for i in range(epochs+1):
           for _, (x,y) in enumerate(dataloader):
               x = x.cuda()
               y = y.cuda()
               out = model(x)
               loss = criterion(out, y)
               loss.backward()
               optmizer.step()


    def evaluate(self, model, dataloader):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        correct = 0
        model.cuda()
        for _, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            if torch.argmax(out) == y:
                correct += 1
        print(correct / len(dataloader))
