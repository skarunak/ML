#!/usr/bin/python

import xlwt
import xlrd
from xlrd.sheet import ctype_text  
import numpy as np
import pprint
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

def GetMeanCov(inp_array):
   mean = np.mean(inp_array, axis=0)

   return mean

def Build2dHist(input_arr_h, bin_h, min_h, max_h, input_arr_hs, bin_hs, min_hs, max_hs):
   out = np.zeros((bin_h, bin_hs), dtype=int)

   for h,hs in zip(input_arr_h, input_arr_hs):
     pos_1 = (int)(math.floor((bin_h-1) * ((float)(h-min_h)/(max_h-min_h))))
     pos_2 = (int)(math.floor((bin_hs-1) * ((float)(hs-min_hs)/(max_hs-min_hs))))
     out[pos_1][pos_2] = out[pos_1][pos_2] + 1

   return out

def PredictGender2dBayes(male_arr, female_arr, h):

   male_heights = []
   male_handspan = []
   female_heights = []
   female_handspan = []

   male_heights.append([list_var[0] for list_var in male_arr])
   male_handspan.append([list_var[1] for list_var in male_arr])
   female_heights.append([list_var[0] for list_var in female_arr])
   female_handspan.append([list_var[1] for list_var in female_arr])

   m_mean = np.mean(male_arr, axis=0)
   f_mean = np.mean(female_arr, axis=0)

   m_cov = np.cov(male_heights, male_handspan)
   f_cov = np.cov(female_heights, female_handspan)
   print "GetMeanCov ..."
   print m_mean
   print f_mean
   print (m_cov)
   print (f_cov)
   print "Male details ..." 
   arg1 = (1/((2*math.pi)*math.sqrt(np.linalg.det(m_cov))))
   arg2 = h - m_mean
   m_cov_arr = np.array(m_cov)
   arg3 = np.linalg.inv(m_cov) 
   arg2_arr = np.matrix(arg2) 
   arg4 = arg2_arr.transpose()
   arg5 = np.matrix(arg2) * np.matrix(arg3)
   print "Details ..."
   print h
   print len(male_arr)
   print arg1
   print arg2
   print arg3
   print arg4 
   print arg5
   print (arg5 * np.matrix(arg4))
   print math.exp((-0.5)*124)
   p_male = len(male_arr) * arg1 *math.exp((-0.5)*(np.matrix(arg2) * np.matrix(arg3) * np.matrix(arg4)))

   arg1 = (1/((2*math.pi)*math.sqrt(np.linalg.det(f_cov))))
   arg2 = (h-f_mean)
   f_cov_arr = np.array(f_cov)
   arg3 = np.linalg.inv(f_cov) 
   arg2_arr = np.matrix(arg2) 
   arg4 = arg2_arr.transpose()
   arg5 = np.matrix(arg2) * np.matrix(arg3)

   p_female = len(male_arr) * arg1 *math.exp((-0.5)*(np.matrix(arg2) * np.matrix(arg3) * np.matrix(arg4)))

   print ("p output %f %f " % (p_male, p_female))

   # Probabilities ..
   prob_male = p_male / (p_male + p_female)
   prob_female = p_female / (p_male + p_female)

   if prob_male > prob_female:
       P = 'Male'
   elif prob_male < prob_female:
       P = 'Female'
   else :
       P = 'Indeterminate'

   print ("Bayes:: Pred: %s p_male: %f p_female: %f" % ( P, prob_male, prob_female))
   
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
male_data = []
female_data = []
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
       male_data.append([height.value, handspan.value])
    else :
       female_heights.append(height.value)
       female_handspan.append(handspan.value)
       female_data.append([height.value, handspan.value])


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

workbook = xlwt.Workbook() 
sheet = workbook.add_sheet("Male")
print male_hist

for row,data in enumerate(male_hist):
	list = data.tolist()
	str_list = str(list).strip('[]')
	sheet.write(row, 0, str_list)
print female_hist

sheet = workbook.add_sheet("Female")
for row,data in enumerate(female_hist):
	list = data.tolist()
	str_list = str(list).strip('[]')
	print row, str_list
#	str = ' '.join(list)
#	print str
	sheet.write(row, 0, str_list)
workbook.save("2dHist.xls") 

test_data = [[69, 17.5], [66,22], [70, 21.5], [69, 23.5]]

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

# PDFs 

print ("Total %d %d %d" % (len(total_heights), min(total_heights), max(total_heights)))
print ("Male %d %d %d" % (len(male_heights), min(male_heights), max(male_heights)))
print ("Female %d %d %d" % (len(female_heights), min(female_heights), max(female_heights)))

for test in test_data:
   PredictGender2dBayes(male_data, female_data, test)

'''
   p_male = len(male_arr) * (1/((2*math.pi)*math.sqrt(np.linalg.det(m_cov))))*math.exp((-0.5)*((h-m_mean) * np.matrix.transpose(m_cov) * np.matrix.transpose(h-m_mean)))
   p_female = len(female_arr) * (1/((2*math.pi)*math.sqrt(np.linalg.det(f_cov))))*math.exp((-0.5)*((h-f_mean) * np.matrix.transpose(f_cov) * np.matrix.transpose(h-f_mean)))
'''
