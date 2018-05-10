# from __future__ import print_function
# import torch
# import torchvision
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import dataobject
# import neuralobject
# from torch.utils.data import DataLoader
# import numpy as np
# import cProfile, pstats
# from io import StringIO
#
# batch_siz = 2
#
# def showpic(foto):
#     print(foto.shape, foto.size, type(foto))
#     plt.imshow(foto)
#     plt.show()
#
# def show_dataset(dataset):
#     for foto, _ in dataset:
#         img = foto / 2 + 0.5  # unnormalize
#         npimg = img.numpy()
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.show()
#
# # trainRaw = dataobject.PersonalDataSet(size=400, rebuild=False, name="train")
# # testRaw = dataobject.PersonalDataSet(size=100, rebuild=False, name="test")
#
# pr = cProfile.Profile()
# pr.enable()
# print(" dataset #1")
# trainRaw = dataobject.PersonalDataSet(size=2000, framesize=32, rebuild=False, name="easy_train", randomness=0.0, grain=0.0)
# # develop = dataobject.PersonalDataSet(size=5, framesize=32, rebuild=True, name="easy_test", randomness=0.0, grain=0.0)
# # show_dataset(develop)
# print(" dataset #2")
# # develop = dataobject.PersonalDataSet(size=2000, framesize=32, rebuild=True, name="middle_train", randomness=0.25, grain=0.15)
# # develop = dataobject.PersonalDataSet(size=5, framesize=32, rebuild=True, name="middle_test", randomness=0.25, grain=0.1)
# # show_dataset(develop)
# print(" dataset #3")
# # trainRaw = dataobject.PersonalDataSetataSet(size=2000, framesize=32, rebuild=False, name="hard_train", randomness=0.75, grain=0.32)
# testRaw = dataobject.PersonalDataSet(size=500, framesize=32, rebuild=False, name="hard_test", randomness=0.75, grain=0.32)
# # show_dataset(develop)
# print(" dataset #4")
# # develop = dataobject.PersonalDataSet(size=6000, framesize=32, rebuild=True, name="hard_complete_train", randomness=0.75, grain=0.32)
# # show_dataset(develop)
# pr.disable()
# s = StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print (s.getvalue())
#
# train = DataLoader(trainRaw, batch_size=batch_siz,
#                                           shuffle=False, num_workers=2)
#
# test = DataLoader(testRaw, batch_size=batch_siz,
#                                           shuffle=False, num_workers=2)
#
#
# classes = ('lijn', 'driehoek', 'parralellogram', 'cirkel')
#
#
#
#
#
#
#
# # 2 + "a"
#
# net = neuralobject.Net()
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# print(len(train), len(trainRaw))
#
# for epoch in range(8):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(train, 0):
#
#         # get the inputs
#         inputs, labels = data
#
#         # wrap them in Variable
#         inputs, labels = Variable(inputs), Variable(labels)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs.float())
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.data[0]
#         if i % 200 == 190:    # print every 200 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 200))
#             running_loss = 0.0
#
#
# print('Finished Training')
#
#
#
#
#
# correct = 0
# total = 0
# for i, data in enumerate(test, 0):
#     # if (i > len(test) - 5):
#     #     break
#     images, labels = data
#     outputs = net(Variable(images).float())
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
#
#
# print("\n\n\n")
#
# class_correct = list(0. for i in range(len(classes)))
# class_total = list(0. for i in range(len(classes)))
# for i, data in enumerate(test, 0):
#     images, labels = data
#     outputs = net(Variable(images).float())
#     _, predicted = torch.max(outputs.data, 1)
#     c = (predicted == labels).squeeze()
#     for i in range(batch_siz):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1
#
#
# for i in range(len(classes)):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))
from dataobject import PersonalDataSet


# automated = [[0.0, 0.0], [0.1, 0.05], [0.3, 0.05], [0.3, 0.05], [0.5, 0.1], [0.7, 0.2], [0.7, 0.2], [0.7, 0.2], [0.7, 0.15], [0.6, 0.1], [0.7, 0.2], [0.7, 0.3], [0.7, 0.25], [0.6, 0.2], [0.6, 0.2], [0.7, 0.3], [0.7, 0.2], [0.7, 0.3], [0.6, 0.35], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]]
#
# cur = [tuple([str(tuple(x))+"_", 1, 0.05]) for x in automated] + [tuple(["hard_train", 20, 0.05])]
#
# print (cur)
#
# # for a in range(0, 80, 10):
# #     for b in range(0, 40, 5):
# #         try:
# #             PersonalDataSet(size=4 * 64, framesize=32, rebuild=False, name=str((a/100.0, b/100.0)) + "_train",
# #                             randomness=a/100.0, grain=b/100.0)
# #         except:
# #             PersonalDataSet(size=4 * 64, framesize=32, rebuild=True, name=str((a/100.0, b/100.0)) + "_train",
# #                             randomness=a / 100.0, grain=b / 100.0)


print(len([1.0, 28.206249999999997, 34.2375, 36.368750000000006, 42.5, 50.118750000000006, 52.506249999999994, 56.81875, 61.44375000000001, 62.31875, 69.20625000000001, 66.00625, 72.92500000000001, 74.69375, 74.69375, 74.90625, 70.55, 71.9, 77.65, 77.75, 79.35, 73.15, 75.8, 80.0, 76.4, 79.9, 81.5, 74.85, 77.65, 81.75, 84.35, 83.2, 81.75, 80.5, 82.9, 84.85, 83.4, 65.05, 83.2, 85.95, 82.85, 84.5, 82.05, 83.75, 83.05, 85.4, 83.2, 82.55, 82.85, 85.1, 85.55, 83.1, 86.55, 82.55, 78.4, 77.5, 84.2, 85.65, 84.4, 84.85, 84.6, 86.75, 87.0, 82.15, 82.35, 85.15, 84.45, 87.85, 84.9, 83.05, 81.8, 84.55, 86.75, 84.3, 86.3, 85.75, 81.4, 85.15, 83.0, 85.85, 82.15, 82.45, 83.2, 84.45, 86.15, 85.85, 85.65, 86.9, 84.6, 87.15, 84.65, 84.0, 84.8, 80.65, 76.45, 85.3, 84.45, 86.7, 81.2, 82.8, 86.05, 87.0, 86.55, 82.2, 84.2, 82.8, 85.95, 84.6, 83.4, 83.15, 87.0, 85.75, 86.0, 82.4, 67.5, 87.15]))