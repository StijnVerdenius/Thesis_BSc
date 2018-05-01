from __future__ import print_function
import matplotlib.pyplot as plt
import dataobject
from torch.utils.data import DataLoader
import numpy as np
from random import randint


class DataBinder(DataLoader):
    def __init__(self, name, batch_size):

        self.name = name
        self.raw = dataobject.PersonalDataSet(rebuild=False, name=self.name)
        super().__init__(self.raw, batch_size=batch_size, shuffle=False, num_workers=2)
        # self.loader = DataLoader(self.raw, batch_size=batch_siz, shuffle=False, num_workers=2)

    def __getitem__(self, index):
        return self.raw[index]

    def showRandomItems(self, numberOfItems = 4):
        for x in range(numberOfItems):
            foto = self.raw[randint(0, len(self.raw)-1)]
            img = foto / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()