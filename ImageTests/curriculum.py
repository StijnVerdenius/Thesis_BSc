from __future__ import print_function
from dataBinder import DataBinder
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


class Curriculum(object):

    def __init__(self, curri, batchSize, testSetName, logFileName, neuralNet,  validationSet,entry = 0, classes= ('lijn', 'driehoek', 'parralellogram', 'cirkel')):
        self.curriJson = curri
        self.batchSize = batchSize
        self.net = neuralNet
        self.classes = classes
        self.batchSize = batchSize
        self.entry = entry
        self.validate = DataBinder(validationSet, self.batchSize)
        self.logFileScore1_1 = open(logFileName + "_score_in_time_loss.csv", "a")
        self.logFileScore1_2 = open(logFileName + "_score_in_time_validationset.csv", "a")
        self.logFileScore2 = open(logFileName + "_score_in_end.csv", "a")
        self.testSet = DataBinder(testSetName, self.batchSize)
        self.data = {}
        for datasetKey, _, _ in self.curriJson:
            self.data[datasetKey] = DataBinder(datasetKey, self.batchSize)

    def doCurricullumForScore(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        scoreCounter_1 = []
        scoreCounter_2 = []

        for datasetKey, epochs, _ in self.curriJson:

            usedDataObject = self.data[datasetKey]

            for epoch in range(epochs):

                running_loss = 0.0
                for i, data in enumerate(usedDataObject, 0):

                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.net(inputs.float())
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.data[0]
                    if i % int(len(usedDataObject)/5) == int((len(usedDataObject)-5)/5):

                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 400))
                        scoreCounter_1.append(running_loss)
                        scoreCounter_2.append(self.doTest(self.validate))
                        running_loss = 0.0


        print('Finished Training')


        self.logFileScore1_1.write(str(self.entry) + "," + str(scoreCounter_1).replace("[", "").replace("]", "\n"))
        self.logFileScore1_2.write(str(self.entry) + "," + str(scoreCounter_2).replace("[", "").replace("]", "\n"))

        percentage_correct = self.doTest(self.testSet)


        print("\n\n\n")

        class_correct = list(0. for i in range(len(self.classes)))
        class_total = list(0. for i in range(len(self.classes)))
        for i, data in enumerate(self.testSet, 0):
            images, labels = data
            outputs = self.net(Variable(images).float())
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(self.batchSize):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

        logEntry = str(self.entry)+","+str(percentage_correct)

        for i in range(len(self.classes)):
            infostr = 'Accuracy of %5s , %2d ' % (
                self.classes[i], 100 * class_correct[i] / class_total[i])
            print(infostr)
            logEntry = logEntry +"," + str(100 * class_correct[i] / class_total[i])

        self.logFileScore2.write(logEntry+"\n")




    def doTest(self, testSet):
        correct = 0
        total = 0
        for i, data in enumerate(testSet, 0):
            images, labels = data
            outputs = self.net(Variable(images).float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the network on the %d test images: %d %%'
              % (
                  len(testSet),(
                100 * correct / total)
              ))

        return (100 * correct / total)