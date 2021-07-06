import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import activation
from torch.nn.modules.pooling import MaxPool1d, MaxPool2d

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class PrintSize(nn.Module):
  def __init__(self):
    super(PrintSize, self).__init__()
    
  def forward(self, x):
    print(x.shape)
    return x

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # First 2D convolutional layer, taking in 3 input channel (image),
      # outputting 64 convolutional features, with a square kernel size of 3
      self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding='same')
      # Second 2D convolutional layer, taking in the 127 input layers,
      # outputting 64 convolutional features, with a square kernel size of 3
      self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding='same')

      # Designed to ensure that adjacent pixels are either all 0s or all active
      # with an input probability
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # First fully connected layer
      self.fc1 = nn.Linear(9216, 128)
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Pass data though conv1
        x =  self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        print(x.shape)
        
def get_model():
    model = nn.Sequential(
            PrintSize(),
            nn.Conv2d(3, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            PrintSize(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.3),
            nn.Flatten(),
            nn.Linear(256*8*8, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 150),
            nn.Softmax(dim=1)
        )

    return model


random_data = torch.rand((1, 3, 256, 256))

my_nn = Net()
# print(my_nn)

# def get_model():
    # input_size_tup = (256, 256, 3)
    # model = Sequential()
    # model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',input_shape=input_size_tup,activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',input_shape=input_size_tup,activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Dropout(rate=0.3))
    # model.add(BatchNormalization())
    # model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',input_shape=input_size_tup,activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',input_shape=input_size_tup,activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(BatchNormalization())
    # model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',input_shape=input_size_tup,activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Dropout(rate=0.3))
    # model.add(Flatten())
    # model.add(Dense(1024,activation='relu'))
    # model.add(Dropout(rate=0.3))
    # model.add(Dense(512,activation='relu'))
    # model.add(Dense(150,activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

    # model.load_weights('CNN_model_weights.h5')
    # return model
    # return 0