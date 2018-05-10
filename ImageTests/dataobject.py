from __future__ import print_function
import torch
from torch.utils.data import Dataset
from random import randint
import pickle
import numpy as np
from figuren import Figuren



class PersonalDataSet(Dataset):

    def generateFigures(self, framesize=32, randomness = 0.0, grain = 0.0):
        inverseRandomness =  randomness
        line, triangle, parralellogram, elipse = [],[],[],[]
        counter = 0
        stopAt = 10

        while (not self.fig.checkPicture(triangle,framesize) or counter > stopAt):
            triangle = self.fig.randomTriangle(pictureSize=framesize, grain=grain, randomTriangle=inverseRandomness)
            counter += 1
        counter = 0
        while (not self.fig.checkPicture(line, framesize) or counter > stopAt):
            line = self.fig.line(pictureSize=framesize, grain=grain, bended=inverseRandomness)
            counter += 1
        counter = 0
        while (not self.fig.checkPicture(elipse, framesize) or counter > stopAt):
            elipse = self.fig.elipse(pictureSize=framesize, grain=grain, elipse=inverseRandomness)
            counter += 1
        counter = 0
        while (not self.fig.checkPicture(parralellogram, framesize) or counter > stopAt):
            parralellogram = self.fig.parralellogram(pictureSize=framesize, grain=grain, parralellogram=inverseRandomness)
            counter += 1
        return line, triangle, parralellogram, elipse

    def __init__(self, size=1000, framesize=32, rebuild=False, name="data", randomness = 0.0, grain = 0.0):

        self.data = []
        self.labels = []
        self.name = name
        self.fig = Figuren()
        if (rebuild):
            print("building datset of size ", size, " randomness:", randomness, " grain:", grain)
            for i in range(size):
                line, triangle, parralellogram, elipse = self.generateFigures(framesize=framesize, randomness=randomness, grain=grain)

                self.data.append(line)
                self.data.append(triangle)
                self.data.append(parralellogram)
                self.data.append(elipse)

                self.labels.append(0)
                self.labels.append(1)
                self.labels.append(2)
                self.labels.append(3)
                # print("generated %d out of %d" % (i,size))

            self.save()
        self.data, self.labels = self.load()

    def save(self):
        objectI = (self.data, self.labels)
        with open(self.name+'Figurendata.pkl', 'wb') as output:
            pickle.dump(objectI, output, pickle.HIGHEST_PROTOCOL)
        print("saved dataset")

    def load(self):
        with open(self.name+'Figurendata.pkl', 'rb') as input:
            return pickle.load(input)

    def __getitem__(self, index):
        plaatje = torch.from_numpy(np.array(self.data[index]))
        labels = self.labels[index]
        return (plaatje, labels)

    def __len__(self):
        return len(self.data)



