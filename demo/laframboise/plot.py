import numpy as np
import matplotlib.pyplot as plt

files = ['potential.txt', 'current.txt','charge.txt']
#files = ['DG0_I2_945/potential.txt', 'DG0_I2_945/current.txt','DG0_I2_945/charge.txt']

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

files = ['num_e.txt', 'num_i.txt']
#files = ['DG0_I2_945/num_e.txt', 'DG0_I2_945/num_i.txt']
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
