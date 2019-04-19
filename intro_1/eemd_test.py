import numpy as np 
from PyEMD import EEMD
from time import time as t

l = 1000
s = np.random.random(l)
eemd = EEMD()
start = t()
s_e = eemd(s)
end = t()

for a in range(len(s)):
    print(f"{s[a]}   {s_e[-1][a]}")
'''
print(s)
print()
print(s_e)
'''
print(s_e.shape)
print(f"EEMD Processing occured in {end-start} seconds, at {l/(end-start)} seconds per index")