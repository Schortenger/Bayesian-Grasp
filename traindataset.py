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
from tac_grasp.resnet import ResNet, Bottleneck, BasicBlock, resnet50, resnet101, resnet152
from tac_grasp.VGGnet import vgg16
import time
import numpy
import pdb
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


class MyDataset(Dataset):
    def __init__(self, txt_path, ignore_zero=True, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        fh = fh.readlines()
        # fh.sort()
        # pdb.set_trace()
        imgs = []
        count0 = count1 = count2 = count3 = 0
        trainsum = 400
        for line in fh:
            line = line.rstrip()
            words = line.split("#", 1)
            # pdb.set_trace()
            if ignore_zero and words[1] == '0' and np.random.random() > 1:  # 0.3 #np.random.random()>0.05:
                # print('.')
                # pdb.set_trace()
                continue
            # if float(words[1])==0:
            #     count0+=1
            #     words[1]='0'
            #     if count0 < trainsum:
            #         imgs.append((words[0], words[1]))
            # # else:
            # #     words[1] = '1'
            # elif 0<float(words[1])<=0.5:
            #     count1 += 1
            #     words[1]='1'
            #     if count1 < trainsum:
            #         imgs.append((words[0], words[1]))
            # elif 0.5<float(words[1])<=0.85:
            #     count2 += 1
            #     words[1] = '2'
            #     if count2 < trainsum:
            #         imgs.append((words[0], words[1]))
            if float(words[1]) <= 0.85:
                words[1] = '0'
                # if count1 < trainsum:
                #     imgs.append((words[0], words[1]))
            elif float(words[1]) > 0.85:
                count3 += 1
                words[1] = '1'
                # if count3 < trainsum:
                #     imgs.append((words[0], words[1]))

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


# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(3*32*32)
            nn.Conv2d(3, 6, 5, 1, 0),  # padding=2 to get the same size
            nn.ReLU(),  # #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
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
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)  # input_size=(3*227*227)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # input_size(96*55*55)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)  # input_size(96*27*27)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # input_size(256*27*27)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)  # input_size(256*13*13)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)  # input_size(384*13*13)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)  # input_size(384*13*13)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)  # input_size(256*13*13)
        self.dense1 = nn.Linear(256 * 6 * 6, 4096)  # input_size(256*6*6)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x = x.view(-1, 256 * 6 * 6)
        x = self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


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

    def __call__(self, image, pointx, pointy):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        cenx = np.int(pointx)
        ceny = np.int(pointy)

        image = image[cenx - new_h / 2: cenx + new_h / 2,
                ceny - new_w / 2: ceny + new_w / 2]

        return image


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize((100, 100)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # trainpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/stage1/trainset.txt'
    # testpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/stage1/testset.txt'
    trainpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/train_dataset.txt'  # unknowntrain.txt'
    testpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/test_dataset.txt'  # unknowntest.txt'
    # testpath = '/home/schortenger/Desktop/IROS/tactile_prior/data/test_dataset.txt'
    # trainset = MyDataset(path,transform=transform)
    trainset = MyDataset(txt_path=trainpath, transform=transform)
    testset = MyDataset(txt_path=testpath, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=20,
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

    pdb.set_trace()
    # img, label = trainset[0]
    # for img, label in trainset:
    #     print (img.size(),label)

    # net = vgg16(True)
    net = resnet50(True)
    # net = resnet101(True)

    # Now load the checkpoint
    # model.load_state_dict(checkpoint)
    # model.eval()

    net.cuda()
    print (net)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # criterion = F.smooth_l1_loss()
    # criterion_test = nn.MSELoss(reduction='none')
    # criterion_test = F.smooth_l1_loss(reduction='none')
    # criterion = torch.nn.L1Loss()
    # criterion_test = torch.nn.L1Loss(reduction='none')
    # optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # train
    print ("training begin")
    AVE = []
    NUM = []
    # for epoch in range(165):
    for epoch in range(165):
        net.train()
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
            # print(label)
            # plt.show()
            # print label
            label = torch.from_numpy(np.array(map(int, label)))
            # print(type(image))
            # print(type(label))

            image = image.cuda()
            label = label.cuda()
            image = Variable(image)
            label = Variable(label)
            # label = label.to(torch.float32)
            # label = label.long()

            # plt.imshow(torchvision.utils.make_grid(image))
            # plt.show()
            # print (label)
            # print (image.shape)
            outputs = net(image)
            outputs = torch.squeeze(outputs, 1)
            # print (outputs)
            # pdb.set_trace()
            loss = criterion(outputs, label)
            # pdb.set_trace()
            # loss = F.smooth_l1_loss(outputs, label)

            # pdb.set_trace()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()  # update the parameters

            running_loss += loss.data
            # pdb.set_trace()

            if i % 2 == 0:
                end = time.time()
                # print ('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s' % (
                # epoch + 1, (i + 1) * 30, running_loss / 100, (end - start)))
                start = time.time()
                running_loss = 0

            epochloss = []
            # pdb.set_trace()
            epochloss.append(loss.data.cpu().numpy())
        epochave = np.mean(epochloss)
        end = time.time()
        print ('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s' % (
            epoch + 1, (i + 1) * 30, epochave, (end - start)))

        NUM.append(epoch + 1)
        AVE.append(epochave)
        print epoch
    # pdb.set_trace()
    plt.cla()
    plt.plot(NUM, AVE, 'b-', lw=3)
    plt.show()

    print ("finish training")

    # test

    correct = 0
    total = 0
    TP = FP = FN = 0

    for data in testloader:

        net.eval()
        # print data
        images, labels = data
        # print labels
        labels = torch.from_numpy(np.array(map(int, labels)))  # tuple to tensor

        images = images.cuda()
        labels = labels.cuda()
        # net.eval()
        outputs = net(Variable(images))
        outputs = torch.squeeze(outputs, 1)
        # labels = labels.to(torch.float32)
        # pdb.set_trace()
        # accuracy = criterion_test(outputs,labels)
        # accuracy = F.smooth_l1_loss(outputs,labels,reduction='none')
        # pdb.set_trace()
        _, predicted = torch.max(outputs, 1)

        # print labels
        total += labels.size(0)
        print 'predicted', predicted.cpu().numpy()
        print 'labels', labels.cpu().numpy()

        epochloss = []
        # pdb.set_trace()
        epochloss.append(loss.data.cpu().numpy())

        # correct += (outputs == labels).sum()

        print total
        labels = labels.cpu().numpy()
        predicted = predicted.cpu().numpy()
        # pdb.set_trace()
        correct += (predicted == labels).sum()

        print correct
        TPnum = 0
        for pre0 in predicted:

            if pre0 == 1 & labels[TPnum] == pre0:
                # pdb.set_trace()
                TP += 1
            TPnum += 1
        FPnum = 0

        for pre1 in predicted:
            # print pre1, labels[FPnum]
            #pdb.set_trace()
            if (pre1 == 1) & (labels[FPnum] != pre1):
                FP += 1

            FPnum += 1
        FNnum = 0

        for pre2 in predicted:
            # print pre2, labels[FNnum]
            #pdb.set_trace()
            if (pre2 == 0) & (labels[FNnum] != pre2):
                FN += 1
            FNnum += 1

        #
        # TP += (predicted == labels & predicted=='1').sum()
        # FP += (predicted != labels & predicted == '1').sum()
        # FN += (predicted != labels & predicted == '0').sum()

    # plt.ion()
    # labels=labels.cpu().numpy()
    # predicted = predicted.cpu().numpy()
    # predictions = outputs.data.cpu().numpy()
    # pdb.set_trace()
    # num = np.arange(1, labels.size+1,1)
    #
    #
    # plt.cla()
    # plt.plot(num, labels,  'r-', lw=3 )
    # plt.plot(num, predicted, 'g-', lw=3)
    #
    # # plt.ioff()
    # plt.show()
    # #pdb.set_trace()

    print 'TP=', TP
    print 'FP=', FP
    print 'FN=', FN

    print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))
    TFP = TP + FP
    print('Precision Rate of the network on the %d test images: %d %%' % (TFP, 100 * TP / TFP))
    TPFN = TP + FN
    print('Recall Rate of the network on the %d test images: %d %%' % (TPFN, 100 * TP / TPFN))

    # img=np.array(img)
    #
    # plt.imshow(img)
    # plt.show()
    # print img
    torch.save(net, '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/net.pkl')
    torch.save(net.state_dict(), '/home/schortenger/Desktop/IROS/tactile_prior/data/datapath/net_params.pkl')
