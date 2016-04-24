#!/usr/bin/python

from __future__ import print_function
from os.path import join, dirname, abspath
import xlrd
from xlrd.sheet import ctype_text  
import numpy as np
import pprint
import math
import matplotlib.pyplot as plt

def BuildHist(input_arr, bin, min_num, max_num):
   out = np.zeros(bin, dtype=int)

   for i,val in enumerate(input_arr):
     pos = (int)(math.ceil((bin-1) * ((float)(input_arr[i]-min_num)/(max_num-min_num))))
     out[pos] = out[pos] + 1

   return out

def PredictGender(h, bin, min_num, max_num, male_hist, female_hist):
   pos = (int)(math.ceil((bin-1) * ((float)(h-min_num)/(max_num-min_num))))

   male_cnt = male_hist[pos]
   female_cnt = female_hist[pos]
   total_cnt = male_cnt + female_cnt

   print ("h: %d ; m_cnt: %d ; f_cnt: %d MP: %f FP: %f" % (h, male_cnt, female_cnt, (float)(male_cnt)/total_cnt, (float)(female_cnt)/total_cnt))

xl_workbook = xlrd.open_workbook('Height_Data.xlsx')

xl_sheet = xl_workbook.sheet_by_index(0)

print ('Sheet name: %s' % xl_sheet.name)

row = xl_sheet.row(0)

for idx, cell_obj in enumerate(row):
    cell_type_str = ctype_text.get(cell_obj.ctype, 'unknown type')
    print('(%s) %s %s' % (idx, cell_type_str, cell_obj.value))

num_cols = xl_sheet.ncols   # Number of columns
female_heights = []
male_heights = []
total_heights = []

for row_idx in range(1, xl_sheet.nrows):    # Iterate through rows
    #print ('Row: %s' % row_idx)   # Print row number
    ft = xl_sheet.cell(row_idx, 0)
    inch = xl_sheet.cell(row_idx, 1)
    gender = xl_sheet.cell(row_idx, 2)

    total_heights.append((ft.value*12)+inch.value)

    if gender.value == 'Male':
       male_heights.append((ft.value*12)+inch.value)
    else :
       female_heights.append((ft.value*12)+inch.value)


min_num = min(total_heights)
max_num = max(total_heights)
bins = 25.0
bin_width = ((float)((max_num-min_num)))/bins
male_hist = np.zeros(25, dtype=int)
female_hist = np.zeros(25, dtype=int)


male_hist = BuildHist(male_heights, bins, min_num, max_num)
female_hist = BuildHist(female_heights, bins, min_num, max_num)

print ("bins %d width %f " % (bins, bin_width))
print ("Total %d %d %d" % (len(total_heights), min(total_heights), max(total_heights)))
print ("Male %d %d %d" % (len(male_heights), min(male_heights), max(male_heights)))
print ("Female %d %d %d" % (len(female_heights), min(female_heights), max(female_heights)))
pprint.pprint (male_hist)
pprint.pprint (female_hist)

input_list = [55, 60, 65, 70, 75, 80]

for h in input_list:
  PredictGender(h, bins, min_num, max_num, male_hist, female_hist)
'''
xaxis = np.arange(min_num, max_num, bin_width)
plt.hist((male_hist, female_hist), bins=xaxis)
plt.xlabel('Height (Inches)')
plt.ylabel('Count')
plt.title('Male Histogram')
plt.grid(True)
plt.show()
    for col_idx in range(0, num_cols):  # Iterate through columns
        cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
        print ('Column: [%s] cell_obj: [%s]' % (col_idx, cell_obj))
'''
