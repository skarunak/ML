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

print "W ..\n" , W

# Convert T2 (nominal -> ordinal)
T2 = np.zeros((n_features, max(T2_list)[0]+1), dtype=int)
print "Shape of T2 " , T2.shape
T2.fill(-1)
n_features = 0
for i in T2_list:
   T2[n_features][i[0]] = 1
   n_features=n_features+1

W_multiclass = PI_Xa * T2
print "W (MC) .. \n", W_multiclass

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
# Binary classifier
    binary = x_list[n_features] * W
# N class classifier ..
    multi = x_list[n_features] * W_multiclass
    print ("%d %d" % (np.sign(binary), multi.argmax()))
    n_features = n_features+1
print "binary classifier : Number of features predicted", n_features

print "On training data .."
# Validate
train_t1 = [] # binary
train_t2 = [] # multi-class

for i in Xa:
    binary = np.sign(i * W)
    train_t1.append(binary[0,0])
    multi = i * W_multiclass
    train_t2.append(multi.argmax())

#print train_t1
#print train_t2

TP=0
TN=0
FP=0
FN=0

for t1,t1_c in zip(T1_list,train_t1):
   if t1[0] == t1_c:
      if t1[0] == -1.0:
         TN = TN + 1
      else:
         TP = TP + 1
   else:
      if t1_c == -1.0:
         FN = FN + 1
      else:
         FP = FP + 1
     
      
print ("TP %d TN %d FP %d FN %d" % (TP, TN, FP, FN))

t2_perf = np.zeros((6,6), dtype=int)
for t2,t2_c in zip(T2_list, train_t2):
   t2_perf[t2[0]][t2_c] = t2_perf[t2[0]][t2_c]+1
print t2_perf
'''
workbook = xlwt.Workbook() 
sheet = workbook.add_sheet("Rslt")
for row,data in enumerate(train_t1):
	str_list = str(data)
	sheet.write(row, 0, str_list)

for row,data in enumerate(train_t2):
	str_list = str(data)
	sheet.write(row, 1, str_list)

workbook.save("LinCls.xls") 
'''
print "Done .."
