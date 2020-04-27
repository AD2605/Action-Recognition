# Description
The aim of this project is to dive into the field of action recognition and explore various techniques.
The model is a pytorch implementation of the paper - **A Closer Look at Spatiotemporal Convolutions for Action Recognition**
Link to the paper is - **https://arxiv.org/abs/1711.11248v3**

# Dependencies 
1 - Opencv to load and resize videos and <br/>
2 - For mixed precision training and fp16 conversions Apex library is used. For installation and more info see https://github.com/NVIDIA/apex.


# To Do 
1 - Implement soft trainable attention and compare the performance. <br/>
2 - Visualize where the model focuses <br/>
3 - Classify actions real time
