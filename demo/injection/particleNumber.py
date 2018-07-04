import numpy as np
import matplotlib.pyplot as plt

files = ['num_e.txt', 'num_i.txt',
         'num_tot.txt', "outside.txt", "injected.txt"]

data = []
for fname in files:
    tmp = []
    f = open(fname, 'r').readlines()
    for j in range(0,len(f)):
        tmp.append(f[j].split()[0])
    tmp = [float(x) for x in tmp]
    data.append(tmp)

num_e = np.array(data[0])
num_i = np.array(data[1])
num_tot = np.array(data[2])

outside = np.array(data[3])
injected = np.array(data[4])

# plt.plot(num_e,label="Number of electrons")
# plt.plot(num_i,label="Number of ions")
# plt.legend(loc='lower right')
# plt.grid()
# plt.xlabel("Timestep")
# plt.ylabel("Number")

plt.figure()
plt.plot(num_tot, label="Total number of particles")
plt.grid()
plt.legend(loc='best')

plt.figure()
plt.plot(injected, label="Particles injected")
plt.plot(outside, label="particles outside")
plt.legend(loc='best')
plt.grid()
plt.xlabel("Timestep")
plt.ylabel("Number")

plt.show()
