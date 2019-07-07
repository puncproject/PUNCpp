from numpy import *
import matplotlib.pyplot as plt

fnames = ["position.txt"]

for j, fname in enumerate(fnames):
    f = open(fname, "r").read()
    tmp = [item.split() for item in f.split('\n')[:-1]]
    pos = array(tmp, dtype=float)
    print(pos.shape)

    fig = plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], label='Particle trajectory')
    plt.xlim([0, 2])
    plt.ylim([0, 2])
    plt.grid()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    #plt.show()

plt.show()
