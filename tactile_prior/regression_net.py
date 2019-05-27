# coding: utf-8
from matplotlib import pyplot as plt
from PIL import Image
import torch, torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tac_grasp.resnet import ResNet, Bottleneck, BasicBlock
import time
import numpy
import pdb
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn.functional as F
import math

class MyDataset(Dataset):
    def __init__(self, txt_path,ignore_zero=True, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        fh = fh.readlines()
        # fh.sort()
        # pdb.set_trace()
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split("#",1)
            # pdb.set_trace()

            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    def __len__(self):
        return len(self.imgs)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential( #input_size=(3*32*32)
        nn.Conv2d(3, 6, 5, 1, 0), #padding=2 to get the same size
        nn.ReLU(),  # #input_size=(6*28*28)
        nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),  # #input_size=(16*10*10)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(16*5*5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

####Alexnet network
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4)#input_size=(3*227*227)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2) #input_size(96*55*55)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)#input_size(96*27*27)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)#input_size(256*27*27)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)#input_size(256*13*13)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)#input_size(384*13*13)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)#input_size(384*13*13)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)#input_size(256*13*13)
        self.dense1 = nn.Linear(256*6*6,4096)#input_size(256*6*6)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096,4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096,1)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x=x.view(-1,256*6*6)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)





class VGG(nn.Module):

    def __init__(self, features, num_classes=1, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.MaxPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 120),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(84, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #pdb.set_trace()
        x = self.features(x)
        #pdb.set_trace()
        x = self.avgpool(x)
        #pdb.set_trace()
        x = x.view(x.size(0), -1)
        #pdb.set_trace()
        x = self.classifier(x)
        #pdb.set_trace()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#https://blog.csdn.net/gbyy42299/article/details/78969261
def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


class PointCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, pointx,pointy):


        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        cenx = np.int(pointx)
        ceny = np.int(pointy)

        image = image[cenx-new_h/2: cenx + new_h/2,
                      ceny-new_w/2: ceny + new_w/2]



        return image





if __name__=='__main__':
    transform = transforms.Compose(
        [transforms.Resize(224),transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #trainpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/train_dataset.txt'
    trainpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/screw/train_dataset.txt'
    testpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/screw/test_dataset.txt'
    #testpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/test_dataset.txt'
    #trainset = MyDataset(path,transform=transform)
    trainset = MyDataset(txt_path=trainpath,transform=transform)
    testset = MyDataset(txt_path=testpath, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=68,
                                             shuffle=True, num_workers=0)

    # debugloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
    # for iii in debugloader:
    #     image, label = iii
    #     # fig = plt.figure()
    #     # aa = int(math.sqrt(image.shape[0])+1)
    #     # for ii in range(image.shape[0]):
    #     #     # pdb.set_trace()
    #     #     ax=fig.add_subplot(aa,aa,ii+1)
    #     plt.imshow(image.cpu().numpy()[0].transpose((1, 2, 0)))
    #     print(label)
    #     plt.show()

    print trainset.__len__()
    #img, label = trainset[0]
    # for img, label in trainset:
    #     print (img.size(),label)

    #net = VGG(make_layers(cfg['D']),num_classes=3).cuda()
    net = resnet50()
    net.cuda()
    print (net)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = F.smooth_l1_loss()
    # criterion_test = nn.MSELoss(reduction='none')
    # criterion_test = F.smooth_l1_loss(reduction='none')
    # criterion = torch.nn.L1Loss()
    # criterion_test = torch.nn.L1Loss(reduction='none')
    # optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9)
    learning_rate=0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # train
    print ("training begin")
    AVE = []
    NUM = []
    for epoch in range(50):
        start = time.time()
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            # print (inputs,labels)
            image, label = data
            # fig = plt.figure()
            # aa = int(math.sqrt(image.shape[0])+1)
            # for ii in range(image.shape[0]):
            #     # pdb.set_trace()
            #     ax=fig.add_subplot(aa,aa,ii+1)
            #     ax.imshow(image.cpu().numpy()[ii].transpose((1, 2, 0)))
            #print(label)
            # plt.show()
            #print label
            label = torch.from_numpy(np.array(map(float,label)))
            label=label.float()
            #print(type(image))
            #print(type(label))

            image = image.cuda()
            label = label.cuda()
            image = Variable(image)
            label = Variable(label)
            # label = label.to(torch.float32)
            # label = label.long()

            #plt.imshow(torchvision.utils.make_grid(image))
            #plt.show()
            #print (label)
            #print (image.shape)
            outputs = net(image)
            outputs = torch.squeeze(outputs, 1)
            #print (outputs)
            # pdb.set_trace()
            loss = criterion(outputs, label)
            # pdb.set_trace()
            # loss = F.smooth_l1_loss(outputs, label)

            # pdb.set_trace()
            optimizer.zero_grad()



            loss.backward()
            optimizer.step() #update the parameters

            running_loss += loss.data
            # pdb.set_trace()

            if i % 2 == 0:
                end = time.time()
                # print ('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s' % (
                # epoch + 1, (i + 1) * 30, running_loss / 100, (end - start)))
                start = time.time()
                running_loss = 0

            epochloss=[]
            #pdb.set_trace()
            epochloss.append(loss.data.cpu().numpy())
        epochave=np.mean(epochloss)
        end = time.time()
        print ('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s' % (
            epoch + 1, (i + 1) * 30, epochave , (end - start)))


        NUM.append(epoch+1)
        AVE.append(epochave)
        print epoch
    #pdb.set_trace()
    plt.cla()
    plt.plot(NUM, AVE, 'b-', lw=3)
    plt.show()




    print ("finish training")

    # test
    net.eval()
    correct = 0
    total = 0
    for data in testloader:
        #print data
        images, labels = data
        labels = torch.from_numpy(np.array(map(float, labels))) #tuple to tensor

        images = images.cuda()
        labels = labels.cuda()
        net.eval()
        outputs = net(Variable(images))
        outputs = torch.squeeze(outputs,1)
        # labels = labels.to(torch.float32)
        #pdb.set_trace()
        # accuracy = criterion_test(outputs,labels)
        # accuracy = F.smooth_l1_loss(outputs,labels,reduction='none')
        #pdb.set_trace()
        #_, predicted = torch.max(outputs, 1)

        #print labels
        total += labels.size(0)
        #print predicted
        #print labels

        # correct += (outputs == labels).sum()
        #correct += (predicted == labels).sum()

    # plt.ion()
    labels=labels.cpu().numpy()
    predictions = outputs.data.cpu().numpy()
    pdb.set_trace()
    num = np.arange(1, labels.size+1,1)


    plt.cla()
    plt.plot(num, labels,  'r-', lw=3 )
    plt.plot(num, predictions, 'g-', lw=3)

    # plt.ioff()
    plt.show()
    #pdb.set_trace()


    # print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))

    # img=np.array(img)
    #
    # plt.imshow(img)
    # plt.show()
    # print img
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_params.pkl')
