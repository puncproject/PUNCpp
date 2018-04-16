import numpy as np
import matplotlib.pyplot as plt

files = ['build/potential.txt', 'build/current.txt','build/charge.txt']

#data = []
for fname in files:
    tmp = []
    f = open(fname, 'r').readlines()
    for j in range(0,len(f)):
        tmp.append(f[j].split()[0])
    tmp = [float(x) for x in tmp]
    #data.append(tmp)

    data = np.array(tmp)

    plt.figure()
    plt.plot(data)
    plt.grid()
    plt.xlabel("Timestep")

files = ['build/num_e.txt', 'build/num_i.txt']

plt.figure()
for fname in files:
    tmp = []
    f = open(fname, 'r').readlines()
    for j in range(0,len(f)):
        tmp.append(f[j].split()[0])
    tmp = [float(x) for x in tmp]
    #data.append(tmp)

    data = np.array(tmp)
    plt.plot(data)
    plt.grid()
    plt.xlabel("Timestep")
    
plt.show()
