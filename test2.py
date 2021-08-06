import numpy as np


comN = 11
comT = 0.5
comv0 = 4.9*comT
h0 = 0.7
a1 = 2*0.4/comT**2
comt = [i for i in range(comN)]
comt = np.array(comt)
comt = 0.5/(comN-1)*comt
h1 = h0 + 0.5*a1*comt*comt
h2 = [h1[-1]+comv0*comt[i] - 0.5*9.8*comt[i]*comt[i] for i in range(comN)]
h3 = h1[::-1]
h1 = np.delete(h1, -1)
h2 = np.delete(h2, -1)
comh = np.hstack((h1,h2,h3))

print(comh.size)
print(h1)
print(h2)
print(h3)
