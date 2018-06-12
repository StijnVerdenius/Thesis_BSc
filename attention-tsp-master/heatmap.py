import matplotlib.pyplot as plt
import numpy as np

import entropy




for entrop in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    cities = 50
    batch = 10000
    # entrop = 0.0

    heatmapsize = 100

    a = entropy.tsp_batch(cities, entrop, batch)
    b = np.array([x.numpy() for x in a])

    incremental = (heatmapsize/(batch/10.0))/(cities*batch)

    output = np.zeros((heatmapsize,heatmapsize))

    for point in b:
        for city in point:
            x = int(round(city[0]*heatmapsize))
            while (x >= heatmapsize):
                x -= 1
            y = int(round(city[1]*heatmapsize))
            while (y >= heatmapsize):
                y -= 1

            # print(x,y)
            output[y][x] += incremental


    print(b.shape)
    plt.imshow(output, cmap='hot', interpolation='nearest')
    plt.show()