from __future__ import print_function
from dataBinder import DataBinder
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import dataobject as dataobject
import neuralobject as neuralobject
from torch.utils.data import DataLoader
import numpy as np
import cProfile, pstats
from io import StringIO
from copy import deepcopy
from dataobject import PersonalDataSet





class Curriculum(object):

    def __init__(self, curri, batchSize, testSetName, logFileName, neuralNet,  validationSet, entry = 0, classes= ('lijn', 'driehoek', 'parralellogram', 'cirkel')):
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


        scoreCounter_1 = []
        scoreCounter_2 = []

        for datasetKey, epochs, _ in self.curriJson:

            usedDataObject = self.data[datasetKey]

            print(usedDataObject.name)

            for epoch in range(epochs):

                running_loss = 0.0
                for i, data in enumerate(usedDataObject, 0):

                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    self.net.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.net(inputs.float())
                    loss = self.net.criterion(outputs, labels)
                    loss.backward()
                    self.net.optimizer.step()

                    # print statistics
                    running_loss += loss.data[0]
                    if i % int(len(usedDataObject)/5) == int((len(usedDataObject)-5)/5):

                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 400))
                        scoreCounter_1.append(running_loss)
                        scoreCounter_2.append(self.doTest(self.validate, self.net))
                        running_loss = 0.0


        print('Finished Training')


        self.logFileScore1_1.write(str(self.entry) + "," + str(scoreCounter_1).replace("[", "").replace("]", "\n"))
        self.logFileScore1_2.write(str(self.entry) + "," + str(scoreCounter_2).replace("[", "").replace("]", "\n"))

        percentage_correct = self.doTest(self.testSet, self.net)


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



    def doEpoch(self, usedDataObject, net,  epoch=0):
        running_loss = 0.0
        for i, data in enumerate(usedDataObject, 0):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            net.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = net.criterion(outputs, labels)
            loss.backward()
            net.optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            # if i % int(len(usedDataObject) / 5) == int((len(usedDataObject) - 5) / 5):
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 400))
            #     # scoreCounter_1.append(running_loss)
            #     # scoreCounter_2.append(self.doTest(self.validate))
            #     running_loss = 0.0



    def doTest(self, testSet, net):
        correct = 0
        total = 0
        for i, data in enumerate(testSet, 0):
            images, labels = data
            outputs = net(Variable(images).float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the network on the %d test images: %d %%'
              % (
                  len(testSet),(
                100 * correct / total)
              ))

        return (100 * correct / total)

    def createCurriculumNavekar(self, score):
        curriculum = []
        achieved = 0
        while(achieved < score):

            curriculum, net = self.recursiveTaskSelect(curriculum, (0.75,0.32), self.net, score)



            achieved = self.doTest(self.validate, net)




        return curriculum

    def recursiveTaskSelect(self, curriculum, dataParam, net, score):
        print("Trying ", dataParam)
        net_ = deepcopy(net)
        task, task_test = self.findTasks(dataParam)
        self.doEpoch(task, net_)
        achieved = self.doTest(task_test, net_)
        print("Score achieved: ", achieved)
        if(achieved>=score):
            print("worked")
            curriculum.append(dataParam)
            return curriculum, net_
        else:
            print("Mislukt, creating sourcetask")
            sourceTask = self.defineSourceTask(dataParam)
            print("source created: ", sourceTask)
            curriculum, net__ = self.recursiveTaskSelect(curriculum, sourceTask, net, score)
            print("terug bij ", dataParam)
            return self.recursiveTaskSelect(curriculum, dataParam, net__, score)

    def defineSourceTask(self, dataParam):
        if (dataParam[0] < 0.01 and dataParam[1] < 0.01):
            return (0.0,0.0)
        if (dataParam[0] > dataParam[1]):
            return (int(dataParam[0]*0.45*100)/100.0, int(dataParam[1]*0.85*100)/100.0)
        else:
            return (int(dataParam[0]*0.45*100)/100.0, int(dataParam[1]*0.85*100)/100.0)

    def findTasks(self, dataParam):
        try:
            rebuild = False
            train_set = PersonalDataSet(size=4*64,framesize=32, rebuild=rebuild, name=str(dataParam)+"_train", randomness=dataParam[0], grain=dataParam[1])
            # test_set = PersonalDataSet(size=64, framesize=32, rebuild=rebuild, name=str((0.0, 0.0))+"_test", randomness=dataParam[0], grain=dataParam[1])
            test_set = 0
            return DataBinder(str(dataParam) + "_train", 4, raw=train_set), DataBinder(str(dataParam) + "_test", 4,
                                                                                       raw=test_set)

        except:
            rebuild = True

            train_set = PersonalDataSet(size=4 * 64, framesize=32, rebuild=rebuild, name=str(dataParam) + "_train",
                                    randomness=dataParam[0], grain=dataParam[1])
            # test_set = PersonalDataSet(size=64, framesize=32, rebuild=rebuild, name=str((0.0, 0.0)) + "_test",
            #                        randomness=dataParam[0], grain=dataParam[1])
            test_set = 0
            return DataBinder(str(dataParam) + "_train", 4, raw=train_set), DataBinder(str(dataParam) + "_test", 4,
                                                                                       raw=test_set)


    def createCurriculumAStar(self, score, limit, target):

        pointers = {}

        net_ = deepcopy(self.net)
        train, _ = self.findTasks((0.0,0.0))
        self.doEpoch(train,net_ )
        achieved = (score-self.doTest(self.validate, net_))

        startpos = [0.0,0.0, 0, achieved, net_, [[0.0,0.0]]]
        stack = [startpos]

        while(True):
            # selectie

            scoreCount = 1000000
            bestPos = -1
            for i, pos in enumerate(stack):
                if ((pos[2]+pos[3]) < scoreCount):
                    bestPos = i
                    scoreCount = pos[2]+pos[3]

            currentPos = stack.pop(bestPos)

            for possiblePos_ in self.PossibleMoves(currentPos, limit, target):
                net__ = deepcopy(currentPos[4])
                train, _ = self.findTasks(tuple(possiblePos_))
                self.doEpoch(train, net__)
                trace = deepcopy(currentPos[5])
                trace.append(possiblePos_)
                print(trace)
                achieved = (score - self.doTest(self.validate, net__))
                pointers[str(possiblePos_)] = str(currentPos[:2])

                newMove = [possiblePos_[0], possiblePos_[1], currentPos[2] + 1, achieved, net__, trace]

                if (achieved < 0):
                    return newMove[5]
                else:
                    stack.append(newMove)

    def PossibleMoves(self, currentPos, limit, target):
        x, y, z = currentPos[0], currentPos[1], currentPos[2] + 1
        if (z < limit):
            res = []
            for a in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                for b in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    ax = a + x
                    by = b + y
                    ax = int(ax*10)/10
                    by = int(by * 100) / 100
                    possibleAX = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                    possibleBY = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

                    if (ax >= 0 and by >= 0 and ax< target[0] and by < target[1]):
                        if (not ax in possibleAX):
                            ax = min(possibleAX, key=lambda x: abs(x - ax))
                        if (not by in possibleBY):
                            by = min(possibleBY, key=lambda x: abs(x - by))
                        res.append([ax, by])
            return res
        else:
            return []

    def doCurricullumAdaptive(self, limit):
        scoreCounter_1 = []
        scoreCounter_2 = []

        epochcounter = 0

        for datasetKey, _, percentage in self.curriJson:

            usedDataObject = self.data[datasetKey]

            print(usedDataObject.name)

            progressbar = [2.0,2.0,2.0,2.0,2.0]
            epoch = 0
            while(np.mean(progressbar) >  percentage):

                epochcounter += 1

                if (epochcounter > limit):
                     break



                running_loss = 0.0
                for i, data in enumerate(usedDataObject, 0):

                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    self.net.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.net(inputs.float())
                    loss = self.net.criterion(outputs, labels)
                    loss.backward()
                    self.net.optimizer.step()

                    # print statistics
                    running_loss += loss.data[0]
                    if i % int(len(usedDataObject) / 5) == int((len(usedDataObject) - 5) / 5):
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 400))
                        newScore = self.doTest(self.validate, self.net)
                        scoreCounter_1.append(running_loss)
                        past = 30.0
                        if len(scoreCounter_2) > 0:
                            past = scoreCounter_2[-1]
                        scoreCounter_2.append(newScore)
                        progressbar.pop(0)

                        progressbar.append((newScore-past)/past)
                        running_loss = 0.0

                epoch += 1
                print("average progress: ", np.mean(progressbar))

            if (epochcounter > limit):
                break


        print('Finished Training')

        self.logFileScore1_1.write(str(self.entry) + "," + str(scoreCounter_1).replace("[", "").replace("]", "\n"))
        self.logFileScore1_2.write(str(self.entry) + "," + str(scoreCounter_2).replace("[", "").replace("]", "\n"))

        percentage_correct = self.doTest(self.testSet, self.net)

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

        logEntry = str(self.entry) + "," + str(percentage_correct)

        for i in range(len(self.classes)):
            infostr = 'Accuracy of %5s , %2d ' % (
                self.classes[i], 100 * class_correct[i] / class_total[i])
            print(infostr)
            logEntry = logEntry + "," + str(100 * class_correct[i] / class_total[i])

        self.logFileScore2.write(logEntry + "\n")




