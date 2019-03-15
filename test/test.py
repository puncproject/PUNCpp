#!/usr/bin/env python

I_true = -0.13579559

import numpy as np
I = []
with open('history.dat') as file:
    for line in file:
        if not line.startswith("#"):
            I.append(float(line.split()[-2]))

I_mean = np.average(I[500:])
rel_err = (I_true-I_mean)/I_mean

print("Comparing true current to average of last part of simulation:")
print("True current:      {:g} A".format(I_true))
print("Simulated current: {:g} A".format(I_mean))
print("Relative error:    {:.2f}%".format(rel_err*100))

if np.abs(rel_err)<0.03:
    print("TEST PASSED (relative error < 3%)")
else:
    print("TEST FAILED (relative error > 3%)")
