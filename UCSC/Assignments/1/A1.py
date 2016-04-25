#!/usr/bin/python

import xlrd
from xlrd.sheet import ctype_text  
import numpy as np
import pprint
import math
import matplotlib.pyplot as plt

def GetMeanSigma(inp_array):
   mean = np.mean(inp_array)
   sigma = np.std(inp_array)

   return mean,sigma

def BuildHist(input_arr, bin, min_num, max_num):
   out = np.zeros(bin, dtype=int)

   for i,val in enumerate(input_arr):
     pos = (int)(math.floor((bin-1) * ((float)(input_arr[i]-min_num)/(max_num-min_num))))
     out[pos] = out[pos] + 1

   return out

def PredictGenderBayes(male_arr, female_arr, h):
   m_mean, m_sigma = GetMeanSigma(male_arr)
   f_mean, f_sigma = GetMeanSigma(female_arr)

   p_male = len(male_arr) * ((1/(math.sqrt(2*math.pi)*m_sigma))*math.exp((-0.5)*((h-m_mean)/m_sigma)**2))
   p_female = len(female_arr) * ((1/(math.sqrt(2*math.pi)*f_sigma))*math.exp((-0.5)*((h-f_mean)/f_sigma)**2))

   if p_male > p_female:
       P = 'Male'
   elif p_male < p_female:
       P = 'Female'
   else :
       P = 'Indeterminate'

   print ("Bayes:: h: %d Pred: %s p_male: %f p_female: %f" % (h, P, p_male, p_female))
   
def PredictGenderHist(h, bin, min_num, max_num, male_hist, female_hist):
   pos = (int)(math.floor((bin-1) * ((float)(h-min_num)/(max_num-min_num))))

   male_cnt = male_hist[pos]
   female_cnt = female_hist[pos]
   total_cnt = male_cnt + female_cnt

   p_male = (float)(male_cnt)/total_cnt
   p_female = (float)(female_cnt)/total_cnt

   if p_male > p_female:
       P = 'Male'
   elif p_male < p_female:
       P = 'Female'
   else :
       P = 'Indeterminate'

   print ("Hist :: h: %d ; Pred: %s m_cnt: %d ; f_cnt: %d MP: %f FP: %f" % (h, P, male_cnt, female_cnt, p_male, p_female))


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
bins = 16  # log N + 1
bin_width = ((float)((max_num-min_num)))/bins
male_hist = np.zeros(bins, dtype=int)
female_hist = np.zeros(bins, dtype=int)


male_hist = BuildHist(male_heights, bins, min_num, max_num)
female_hist = BuildHist(female_heights, bins, min_num, max_num)

# PDFs 

male_mean, male_sigma = GetMeanSigma(male_heights)
female_mean, female_sigma = GetMeanSigma(female_heights)

print ("bins %d width %f MMu: %f MSig: %f FMu: %f FSig: %f" % (bins, bin_width, male_mean, male_sigma, female_mean, female_sigma))
print ("Total %d %d %d" % (len(total_heights), min(total_heights), max(total_heights)))
print ("Male %d %d %d" % (len(male_heights), min(male_heights), max(male_heights)))
print ("Female %d %d %d" % (len(female_heights), min(female_heights), max(female_heights)))
pprint.pprint (male_hist)
pprint.pprint (female_hist)

input_list = [55, 60, 65, 70, 75, 80]

for h in input_list:
  PredictGenderHist(h, bins, min_num, max_num, male_hist, female_hist)
  PredictGenderBayes(male_heights, female_heights, h)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist((male_heights, female_heights), bins)
plt.show()

'''
bins = 30
bin_width = ((float)((max_num-min_num)))/bins
male_hist = np.zeros(bins, dtype=int)
female_hist = np.zeros(bins, dtype=int)


male_hist = BuildHist(male_heights, bins, min_num, max_num)
female_hist = BuildHist(female_heights, bins, min_num, max_num)

print ("bins %d width %f " % (bins, bin_width))
print ("Total %d %d %d" % (len(total_heights), min(total_heights), max(total_heights)))
print ("Male %d %d %d" % (len(male_heights), min(male_heights), max(male_heights)))
print ("Female %d %d %d" % (len(female_heights), min(female_heights), max(female_heights)))

pprint.pprint (male_hist)
pprint.pprint (female_hist)

for h in input_list:
  PredictGenderHist(h, bins, min_num, max_num, male_hist, female_hist)
'''
# Limited data
bins = 8
total_heights = [] 
male_heights = [] 
female_heights = []

for row_idx in range(1, 201):    # Iterate through rows
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
bin_width = ((float)((max_num-min_num)))/bins
male_hist = np.zeros(bins, dtype=int)
female_hist = np.zeros(bins, dtype=int)


male_hist = BuildHist(male_heights, bins, min_num, max_num)
female_hist = BuildHist(female_heights, bins, min_num, max_num)

male_mean, male_sigma = GetMeanSigma(male_heights)
female_mean, female_sigma = GetMeanSigma(female_heights)

print ("TRUNCATED !! bins %d width %f MMu: %f MSig: %f FMu: %f FSig: %f" % (bins, bin_width, male_mean, male_sigma, female_mean, female_sigma))
print ("Total %d %d %d" % (len(total_heights), min(total_heights), max(total_heights)))
print ("Male %d %d %d" % (len(male_heights), min(male_heights), max(male_heights)))
print ("Female %d %d %d" % (len(female_heights), min(female_heights), max(female_heights)))
pprint.pprint (male_hist)
pprint.pprint (female_hist)

input_list = [55, 60, 65, 70, 75, 80]

for h in input_list:
  PredictGenderHist(h, bins, min_num, max_num, male_hist, female_hist)
  PredictGenderBayes(male_heights, female_heights, h)

'''
xaxis = np.arange(min_num, max_num, bin_width)
plt.hist([male_hist, female_hist], bins=xaxis, histtype='bar')
plt.xlabel('Height (Inches)')
plt.ylabel('Count')
plt.title('Male Histogram')
plt.grid(True)
plt.show()
    for col_idx in range(0, num_cols):  # Iterate through columns
        cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
        print ('Column: [%s] cell_obj: [%s]' % (col_idx, cell_obj))
'''
