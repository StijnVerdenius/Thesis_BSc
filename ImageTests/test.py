from __future__ import print_function
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import dataobject
import neuralobject
from torch.utils.data import DataLoader
import numpy as np
import cProfile, pstats
from io import StringIO

batch_siz = 2

def showpic(foto):
    print(foto.shape, foto.size, type(foto))
    plt.imshow(foto)
    plt.show()

def show_dataset(dataset):
    for foto, _ in dataset:
        img = foto / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

# trainRaw = dataobject.PersonalDataSet(size=400, rebuild=False, name="train")
# testRaw = dataobject.PersonalDataSet(size=100, rebuild=False, name="test")

pr = cProfile.Profile()
pr.enable()
print(" dataset #1")
trainRaw = dataobject.PersonalDataSet(size=2000, framesize=32, rebuild=False, name="easy_train", randomness=0.0, grain=0.0)
# develop = dataobject.PersonalDataSet(size=5, framesize=32, rebuild=True, name="easy_test", randomness=0.0, grain=0.0)
# show_dataset(develop)
print(" dataset #2")
# develop = dataobject.PersonalDataSet(size=2000, framesize=32, rebuild=True, name="middle_train", randomness=0.25, grain=0.15)
# develop = dataobject.PersonalDataSet(size=5, framesize=32, rebuild=True, name="middle_test", randomness=0.25, grain=0.1)
# show_dataset(develop)
print(" dataset #3")
# trainRaw = dataobject.PersonalDataSetataSet(size=2000, framesize=32, rebuild=False, name="hard_train", randomness=0.75, grain=0.32)
testRaw = dataobject.PersonalDataSet(size=500, framesize=32, rebuild=False, name="hard_test", randomness=0.75, grain=0.32)
# show_dataset(develop)
print(" dataset #4")
# develop = dataobject.PersonalDataSet(size=6000, framesize=32, rebuild=True, name="hard_complete_train", randomness=0.75, grain=0.32)
# show_dataset(develop)
pr.disable()
s = StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print (s.getvalue())

train = DataLoader(trainRaw, batch_size=batch_siz,
                                          shuffle=False, num_workers=2)

test = DataLoader(testRaw, batch_size=batch_siz,
                                          shuffle=False, num_workers=2)


classes = ('lijn', 'driehoek', 'parralellogram', 'cirkel')







# 2 + "a"

net = neuralobject.Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print(len(train), len(trainRaw))

for epoch in range(8):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train, 0):

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 200 == 190:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0


print('Finished Training')





correct = 0
total = 0
for i, data in enumerate(test, 0):
    # if (i > len(test) - 5):
    #     break
    images, labels = data
    outputs = net(Variable(images).float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


print("\n\n\n")

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
for i, data in enumerate(test, 0):
    images, labels = data
    outputs = net(Variable(images).float())
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(batch_siz):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))