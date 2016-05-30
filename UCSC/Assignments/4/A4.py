#!/usr/bin/python

import xlwt
import xlrd
from xlrd.sheet import ctype_text  
import numpy as np
import pprint
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

xl_workbook = xlrd.open_workbook('Stability_Assignment.xlsx')

xl_sheet = xl_workbook.sheet_by_index(0)

print ('Sheet name: %s' % xl_sheet.name)

row = xl_sheet.row(1)

print "row 0 len: " , xl_sheet.row_len(1)
print type(xl_sheet.row_slice(1, 0, 15))

'''
for idx, cell_obj in enumerate(row):
    cell_type_str = ctype_text.get(cell_obj.ctype, 'unknown type')
    print('(%s) %s %s' % (idx, cell_type_str, cell_obj.value))
'''
X_list = []
Xa_list = []
T1_list = []
T2_list = []
offset = 1
n_features = 0

for row_idx in range(1, xl_sheet.nrows):    # Iterate through rows
    #print ('Row: %s' % row_idx)   # Print row number
    X_list.append(xl_sheet.row_values(row_idx, 0, 15))
    Xa_list.append(xl_sheet.row_values(row_idx, 0, 15))
    Xa_list[n_features].insert(0, offset)
    T1_list.append(xl_sheet.row_values(row_idx,15,16))
    T2_list.append(xl_sheet.row_values(row_idx,16,17))

    n_features = n_features+1

print "Number of features ", n_features

T1 = np.array(T1_list)
#T1 = np.reshape(T1, (1, n_features))
Xa = np.matrix(Xa_list)
PI_Xa = np.linalg.pinv(Xa)

W = PI_Xa * T1

#print "W ..\n" , W

# Convert T2 (nominal -> ordinal)
T2 = np.zeros((n_features, max(T2_list)[0]+1), dtype=int)
print "Shape of T2 " , T2.shape
T2.fill(-1)
n_features = 0
for i in T2_list:
   T2[n_features][i[0]] = 1
   n_features=n_features+1

xl_sheet = xl_workbook.sheet_by_index(2)

# Testing set ...
print ('Sheet name: %s' % xl_sheet.name)

'''
row = xl_sheet.row(4)
for idx, cell_obj in enumerate(row):
    cell_type_str = ctype_text.get(cell_obj.ctype, 'unknown type')
    print('(%s) %s %s' % (idx, cell_type_str, cell_obj.value))

'''
x_list = []
n_features = 0

#Binary classifier ..
for row_idx in range(4, xl_sheet.nrows):    # Iterate through rows
    x_list.append(xl_sheet.row_values(row_idx, 0, 15))
    x_list[n_features].insert(0, offset)
    print np.sign(x_list[n_features] * W)
    n_features = n_features+1
print "binary classifier : Number of features predicted", n_features

# N class classifier ..


print "Done .."
