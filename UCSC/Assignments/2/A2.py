#!/usr/bin/python

import xlrd
from xlrd.sheet import ctype_text  
import numpy as np
import pprint
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

def GetMeanSigma(inp_array):
   mean = np.mean(inp_array)
   sigma = np.std(inp_array)

   return mean,sigma

def Build2dHist(input_arr_h, bin_h, min_h, max_h, input_arr_hs, bin_hs, min_hs, max_hs):
   out = np.zeros((bin_h, bin_hs), dtype=int)

   for h,hs in zip(input_arr_h, input_arr_hs):
     pos_1 = (int)(math.floor((bin_h-1) * ((float)(h-min_h)/(max_h-min_h))))
     pos_2 = (int)(math.floor((bin_hs-1) * ((float)(hs-min_hs)/(max_hs-min_hs))))
     out[pos_1][pos_2] = out[pos_1][pos_2] + 1

   return out

def PredictGenderBayes(male_arr, female_arr, h):
   m_mean, m_sigma = GetMeanSigma(male_arr)
   f_mean, f_sigma = GetMeanSigma(female_arr)

   p_male = len(male_arr) * ((1/(math.sqrt(2*math.pi)*m_sigma))*math.exp((-0.5)*((h-m_mean)/m_sigma)**2))
   p_female = len(female_arr) * ((1/(math.sqrt(2*math.pi)*f_sigma))*math.exp((-0.5)*((h-f_mean)/f_sigma)**2))

   # Probabilities ..
   prob_male = p_male / (p_male + p_female)
   prob_female = p_female / (p_male + p_female)

   if prob_male > prob_female:
       P = 'Male'
   elif prob_male < prob_female:
       P = 'Female'
   else :
       P = 'Indeterminate'

   print ("Bayes:: h: %d Pred: %s p_male: %f p_female: %f" % (h, P, prob_male, prob_female))
   
def PredictGender2dHist(h, hs, male_hist, bin_h, min_h, max_h, female_hist, bin_hs, min_hs, max_hs):
   pos_h = (int)(math.floor((bin_h-1) * ((float)(h-min_h)/(max_h-min_h))))
   pos_hs = (int)(math.floor((bin_hs-1) * ((float)(hs-min_hs)/(max_hs-min_hs))))

   male_cnt = male_hist[pos_h][pos_hs]
   female_cnt = female_hist[pos_h][pos_hs]
   total_cnt = male_cnt + female_cnt

   if male_cnt != 0:
     p_male = (float)(male_cnt)/total_cnt
   else :
     p_male = 0
   
   if female_cnt != 0:
     p_female = (float)(female_cnt)/total_cnt
   else :
     p_female = 0

   if p_male > p_female:
       P = 'Male'
   elif p_male < p_female:
       P = 'Female'
   else :
       P = 'Indeterminate'

   print ("Hist :: h: %d hs: %d ; pos: %d,%d Pred: %s m_cnt: %d ; f_cnt: %d MP: %f FP: %f" % (h, hs, pos_h, pos_hs, P, male_cnt, female_cnt, p_male, p_female))


xl_workbook = xlrd.open_workbook('Height_Handspan.xlsx')

xl_sheet = xl_workbook.sheet_by_index(0)

print ('Sheet name: %s' % xl_sheet.name)

row = xl_sheet.row(0)

for idx, cell_obj in enumerate(row):
    cell_type_str = ctype_text.get(cell_obj.ctype, 'unknown type')
    print('(%s) %s %s' % (idx, cell_type_str, cell_obj.value))

num_cols = xl_sheet.ncols   # Number of columns
female_heights = []
male_heights = []
female_handspan = []
male_handspan = []
total_heights = []
total_handspan = []

for row_idx in range(1, xl_sheet.nrows):    # Iterate through rows
    #print ('Row: %s' % row_idx)   # Print row number
    gender = xl_sheet.cell(row_idx, 0)
    height = xl_sheet.cell(row_idx, 1)
    handspan = xl_sheet.cell(row_idx, 2)

    total_heights.append(height.value)
    total_handspan.append(handspan.value)

    if gender.value == "'Male'":
       male_heights.append(height.value)
       male_handspan.append(handspan.value)
    else :
       female_heights.append(height.value)
       female_handspan.append(handspan.value)


min_height = min(total_heights)
max_height = max(total_heights)
min_hs = min(total_handspan)
max_hs = max(total_handspan)

bin_h = 22
bin_hs = 10

bin_h_width = ((float)((max_height-min_height)))/bin_h
bin_hs_width = ((float)((max_hs-min_hs)))/bin_hs

print ("Min h: %f hs: %f ; Max h: %f hs: %f b_h_w: %f b_hs_w: %f " % (min_height, min_hs, max_height, max_hs, bin_h_width, bin_hs_width))

attrib = {}
attrib['min_height'] = min_height
attrib['max_height'] = max_height
attrib['min_hs'] = min_hs
attrib['max_hs'] = max_hs
attrib['bin_h'] = bin_h
attrib['bin_hs'] = bin_hs

male_hist = np.zeros((bin_h, bin_hs), dtype=int)
female_hist = np.zeros((bin_h, bin_hs), dtype=int)

male_hist = Build2dHist(male_heights, bin_h, min_height, max_height, male_handspan, bin_hs, min_hs, max_hs)
female_hist = Build2dHist(female_heights, bin_h, min_height, max_height, female_handspan, bin_hs, min_hs, max_hs)

print male_hist
print female_hist

test_data = [(69, 17.5), (66,22), (70, 21.5), (69, 23.5)]

for h,hs in test_data:
   PredictGender2dHist(h, hs, male_hist, bin_h, min_height, max_height, female_hist, bin_hs, min_hs, max_hs)

'''
xedges = range(22)
yedges = range(10)

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131)
ax.set_title('imshow: equidistant')
im = plt.imshow(male_hist, interpolation='nearest', origin='low',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
'''
'''
fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(male_hist)
ax.set_aspect('equal')
cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()
'''
'''
# Find bin size for height & HS and form a 2D array
bins = 32  # max-min with 1inch resolution
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

input_list = [55, 60, 65, 70, 75, 79, 80]

for h in input_list:
  PredictGenderHist(h, bins, min_num, max_num, male_hist, female_hist)
  PredictGenderBayes(male_heights, female_heights, h)

# Limited data
bins = 25
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

input_list = [55, 60, 65, 70, 75, 79, 80]

for h in input_list:
  PredictGenderHist(h, bins, min_num, max_num, male_hist, female_hist)
  PredictGenderBayes(male_heights, female_heights, h)
'''
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
