import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import time

# import matplotlib.pyplot as plt

torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
if_use_gpu = 1

# accquire training data
training_data = torchvision.datasets.MNIST(
    root='./mnist/',  # saving path of dataset
    train=True, #True: traing dataset,False: test dataset
    transform=torchvision.transforms.ToTensor(),  # normalization to (0,1)
    download=DOWNLOAD_MNIST,
)


print(training_data.train_data.size())
print(training_data.train_labels.size())
# torch.Size([60000, 28, 28])
# torch.Size([60000])

# plt.imshow(training_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % training_data.train_labels[0])
# plt.show()

# The dataset format obtained by torchvision.datasets can be directly placed in the DataLoader
train_loader = Data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE,
                               shuffle=True)

# accquire test dataset

test_data = torchvision.datasets.MNIST(
    root='./mnist/',  # saving path of the test dataset
    train=False,
    transform=torchvision.transforms.ToTensor(),  # normalization
    download=DOWNLOAD_MNIST,
)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1).float(), requires_grad=False)
# test_x = test_x.cuda()
## (~, 28, 28) to (~, 1, 28, 28), in range(0,1)
test_y = test_data.test_labels


# test_y = test_y.cuda()
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
        self.conv1 = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=1, padding=2),  # (16,28,28)
            # get the same size image, padding=(kernel_size-1)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,14,14)
        )
        self.conv2 = nn.Sequential(  # (16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten (batch, 32, 7, 7) to (batch, 32*7*7)
        output = self.out(x)
        return output


cnn = CNN()
if if_use_gpu:
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    start = time.time()
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch:', epoch, '|Step:', step,
                  '|train loss:%.4f' % loss.data[0])
    duration = time.time() - start
    print('Training duation: %.4f' % duration)

cnn = cnn.cpu()
test_output = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = sum(pred_y == test_y) / test_y.size(0)
print('Test Acc: %.4f' % accuracy)