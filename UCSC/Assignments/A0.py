#!/usr/bin/python
import numpy as np
import math 

def BuildHist(input_arr, bin):
   out = np.zeros(bin, dtype=int)

   min_num = min(input_arr)
   max_num = max(input_arr)

   for i in input_arr:
     pos = (int)(math.ceil((bin-1) * ((float)(max_num - input_arr[i])/max_num)))
     out[pos] = out[pos] + 1

   return out

bin = 10
inp = list(range(100))
hist = BuildHist(inp, bin)
print hist
