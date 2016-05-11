#!/usr/bin/python
import os
import struct
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
def BuildGender2dBayes(male_arr, female_arr):

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
   #print "GetMeanCov ..."
   #print m_mean
   #print f_mean
   #print (m_cov)
   #print (f_cov)
   #print "Male details ..." 
   m_first = (1/((2*math.pi)*math.sqrt(np.linalg.det(m_cov))))
   #arg2 = h - m_mean
   #m_cov_arr = np.array(m_cov)
   #arg3 = np.linalg.inv(m_cov) 
   #arg2_arr = np.matrix(arg2) 
   #arg4 = arg2_arr.transpose()
   #arg5 = np.matrix(arg2) * np.matrix(arg3)
   #print "Details ..."
   #print h
   #print len(male_arr)
   #print arg1
   #print arg2
   #print arg3
   #print arg4 
   #print arg5
   #print (arg5 * np.matrix(arg4))
   #print math.exp((-0.5)*124)
   #p_male = len(male_arr) * arg1 *math.exp((-0.5)*(np.matrix(arg2) * np.matrix(arg3) * np.matrix(arg4)))

   f_first = (1/((2*math.pi)*math.sqrt(np.linalg.det(f_cov))))
   #arg2 = (h-f_mean)
   #f_cov_arr = np.array(f_cov)
   #arg3 = np.linalg.inv(f_cov) 
   #arg2_arr = np.matrix(arg2) 
   #arg4 = arg2_arr.transpose()
   #arg5 = np.matrix(arg2) * np.matrix(arg3)

   #p_female = len(male_arr) * arg1 *math.exp((-0.5)*(np.matrix(arg2) * np.matrix(arg3) * np.matrix(arg4)))

   #print ("p output %f %f " % (p_male, p_female))

   # Probabilities ..
   #prob_male = p_male / (p_male + p_female)
   #prob_female = p_female / (p_male + p_female)

   return m_mean, m_cov, m_first, f_mean, f_cov, f_first

def PredictGender2dBayes(male_arr, female_arr, h, m_mean, m_cov, m_first, f_mean, f_cov, f_first):

   #male_heights = []
   #male_handspan = []
   #female_heights = []
   #female_handspan = []

   #male_heights.append([list_var[0] for list_var in male_arr])
   #male_handspan.append([list_var[1] for list_var in male_arr])
   #female_heights.append([list_var[0] for list_var in female_arr])
   #female_handspan.append([list_var[1] for list_var in female_arr])

   #m_mean = np.mean(male_arr, axis=0)
   #f_mean = np.mean(female_arr, axis=0)

   #m_cov = np.cov(male_heights, male_handspan)
   #f_cov = np.cov(female_heights, female_handspan)
   #print "GetMeanCov ..."
   #print m_mean
   #print f_mean
   #print (m_cov)
   #print (f_cov)
   #print "Male details ..." 
   arg1 = (1/((2*math.pi)*math.sqrt(np.linalg.det(m_cov))))
   arg2 = h - m_mean
   m_cov_arr = np.array(m_cov)
   arg3 = np.linalg.inv(m_cov) 
   arg2_arr = np.matrix(arg2) 
   arg4 = arg2_arr.transpose()
   arg5 = np.matrix(arg2) * np.matrix(arg3)
   #print "Details ..."
   #print h
   #print len(male_arr)
   #print arg1
   #print arg2
   #print arg3
   #print arg4 
   #print arg5
   #print (arg5 * np.matrix(arg4))
   #print math.exp((-0.5)*124)
   p_male = len(male_arr) * arg1 *math.exp((-0.5)*(np.matrix(arg2) * np.matrix(arg3) * np.matrix(arg4)))

   arg1 = (1/((2*math.pi)*math.sqrt(np.linalg.det(f_cov))))
   arg2 = (h-f_mean)
   f_cov_arr = np.array(f_cov)
   arg3 = np.linalg.inv(f_cov) 
   arg2_arr = np.matrix(arg2) 
   arg4 = arg2_arr.transpose()
   arg5 = np.matrix(arg2) * np.matrix(arg3)

   p_female = len(male_arr) * arg1 *math.exp((-0.5)*(np.matrix(arg2) * np.matrix(arg3) * np.matrix(arg4)))

   #print ("p output %f %f " % (p_male, p_female))

   # Probabilities ..
   prob_male = p_male / (p_male + p_female)
   prob_female = p_female / (p_male + p_female)

   if prob_male > prob_female:
       P = 'Male'
   elif prob_male < prob_female:
       P = 'Female'
   else :
       P = 'Indeterminate'

   #print ("Bayes:: Pred: %s p_male: %f p_female: %f" % ( P, prob_male, prob_female))

   return prob_male, prob_female

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

   #print ("Hist :: h: %d hs: %d ; pos: %d,%d Pred: %s m_cnt: %d ; f_cnt: %d MP: %f FP: %f" % (h, hs, pos_h, pos_hs, P, male_cnt, female_cnt, p_male, p_female))
   return p_male, p_female

def Build2dHist(input_arr, bin_h, min_h, max_h, bin_hs, min_hs, max_hs):
   out = np.zeros((bin_h, bin_hs), dtype=int)

   for h,hs in input_arr:
     pos_1 = (int)(math.floor((bin_h-1) * ((float)(h-min_h)/(max_h-min_h))))
     pos_2 = (int)(math.floor((bin_hs-1) * ((float)(hs-min_hs)/(max_hs-min_hs))))
     out[pos_1][pos_2] = out[pos_1][pos_2] + 1

   return out

def read(dataset = "testing", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    print "Labels: mg %d num %d" % (magic, num)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    
    print "Images: mg %d num %d rows %d cols %d" % (magic, num, rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx].ravel())

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def write_image_to_text_file(image_array, digit_name):
    flat_arr = image_array.ravel()
    digt_image_text = os.path.join(".", digit_name+'_image_text')
    fdesc = open (digt_image_text, "w")
    for count in range (len(flat_arr)):
        #fdesc.write(str(image_array[count]) + "\n")
        fdesc.write(str(int(math.floor(abs(flat_arr[count])))) + "\n")
    else :
        print ("Done with image to text operation")
    fdesc.close()

def write_cov_to_text_file(arr):
    fdesc = open ("image_cov.txt", "w")
    for row in range (len(arr)):
        for col in range(len(arr)):
            fdesc.write(str(int(math.floor(abs(arr[row][col])))) + "\n")
    else :
        print ("Done with cov-matrix to text operation")
    fdesc.close()

def get_lables(dataset, lbl1, lbl2, lbl3):
    combined = []
    first = []
    second = []
    third = []
    for lbl, vec in read(dataset):
        if (lbl == lbl1 or lbl == lbl2):
           combined.append((lbl, vec))
        else:
           third.append((lbl, vec))

        if (lbl == lbl1):
           first.append((lbl, vec))
        if (lbl == lbl2):
           second.append((lbl, vec))
        #if (lbl == lbl3):
        #   third.append((lbl, vec))
    return combined,first,second,third

def getp1p2oftest(img, mean, V):
    z = img - mean
    zVT = V[:2,:].transpose()
    zp1p2 = z * zVT
    return zp1p2

def predictlable(classifier, test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max):
   correct = 0
   wrong = 0
   indet = 0

   if classifier is 'gaus':
       m_mean, m_cov, m_first, f_mean, f_cov, f_first = BuildGender2dBayes(P1P2_dig1, P1P2_dig2)

   for lbl, vec in test_inp:
     test_p1p2 = getp1p2oftest(vec, mean, V)
     if classifier is 'hist':
       p_dig1 , p_dig2 = PredictGender2dHist(test_p1p2[0,0], test_p1p2[0,1], dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
     else:
       p_dig1, p_dig2 = PredictGender2dBayes(P1P2_dig1, P1P2_dig2, test_p1p2, m_mean, m_cov, m_first, f_mean, f_cov, f_first)

     if p_dig1 > p_dig2:
        out = first
     elif p_dig1 < p_dig2:
        out = second
     else :
        out = -1

     if (out == lbl):
        correct = correct + 1
     elif (out == -1):
        indet = indet + 1
     else:
        wrong = wrong + 1
   
   return correct, wrong, indet

#Filter 3 & 4

lbl_img_list = [] # (lable, image) sorted by lable
lbl1 = []
lbl2 = []
lbl3 = []
first = 3
second = 4
third = 1

lbl_img_list, lbl1, lbl2, lbl3 = get_lables('training', first,second, third)
'''
for lbl,vec in read():
   if (lbl == 3 or lbl == 4):
      lbl_img_list.append((lbl, vec))
      if (lbl == 3):
         lbl1.append((lbl, vec))
      if (lbl == 4):
         lbl2.append((lbl, vec))
'''
lbl_img_list = sorted(lbl_img_list, key=lambda key: key[0])
print ("Data set: sz %d " % (len(lbl_img_list)))

X_arr = np.stack((j for i,j in lbl_img_list))

#print "X array " , X_arr[0]
mean = np.mean(X_arr, axis=0)
#print "mean array ", mean

#plt.plot(mean)
#plt.show()

Z = X_arr - mean
z_row, z_col = Z.shape
print ("Z : (%d, %d) " % (z_row, z_col))

# Mean of Z matrix
#m_Z = np.mean(Z, axis=0)
#print "m_Z :: ", m_Z[0]

# cov. of Z
C = np.cov(Z, rowvar=False)
cov_row, cov_col = C.shape
print ("C : (%d, %d) " % (cov_row, cov_col))
# Write image file for Cov
#write_cov_to_text_file(C)

# Eigen 
eg_val, V = np.linalg.eig(C)
#print ("Eig.Val : (%d, %d) " % (eg_val[0].real, eg_val[1].real))
V = V.real
V_row, V_col = V.shape
print ("V : (%d, %d) " % (V_row, V_col))

# P
V = np.matrix(V).transpose()
TV = V.transpose()
P = Z * TV
P_mean = np.mean(P, axis=0)
#print "P_mean ..", P_mean

P1P2 = P[:,0:2]
P1P2_arr = np.array(P1P2)

P1P2_dig1 = P1P2_arr[:len(lbl1), 0:2]
P1P2_dig2 = P1P2_arr[len(lbl1):, 0:2]

#scatter plot
f, ax = plt.subplots()
ax.scatter (P1P2_dig1[:,0], P1P2_dig1[:,1], c='r', marker='+')
ax.scatter (P1P2_dig2[:,0], P1P2_dig2[:,1], c='b', marker='+')
ax.set_xlabel("p1")
ax.set_ylabel("p2")
#plt.show()

# Recontruction with loss
R_ = P1P2 * V[:2,:]
X_ = R_ + mean

write_image_to_text_file(np.array(X_[0]), 'rec_1_'+str(lbl_img_list[0][0]))
write_image_to_text_file(np.array(X_[1]), 'rec_2_'+ str(lbl_img_list[1][0]))
write_image_to_text_file(np.array(X_[1200]), 'rec_3_'+str(lbl_img_list[1200][0]))
write_image_to_text_file(np.array(X_[1201]), 'rec_4_'+str(lbl_img_list[1201][0]))

write_image_to_text_file(np.array(X_arr[0]), 'orig_1_'+str(lbl_img_list[0][0]))
write_image_to_text_file(np.array(X_arr[1]), 'orig_2_'+ str(lbl_img_list[1][0]))
write_image_to_text_file(np.array(X_arr[1200]), 'orig_3_'+str(lbl_img_list[1200][0]))
write_image_to_text_file(np.array(X_arr[1201]), 'orig_4_'+str(lbl_img_list[1201][0]))
# Loss less reconstruction
R = P * V

p1_max = max(P1P2[:,0])
p1_min = min(P1P2[:,0])
p1_bin = math.ceil(np.log2(len(P1P2)))+1

p2_max = max(P1P2[:,1])
p2_min = min(P1P2[:,1])
p2_bin = p1_bin

print ("Hist:: P1: %d,%d,%d P2: %d,%d,%d" %(p1_max, p1_min, p1_bin, p2_max, p2_min, p2_bin))

dig1_hist = Build2dHist(P1P2_dig1, p1_bin, p1_min, p1_max, p2_bin, p2_min, p2_max)
dig2_hist = Build2dHist(P1P2_dig2, p1_bin, p1_min, p1_max, p2_bin, p2_min, p2_max)

# Plot histogram ??

test_lbl_img_list = [] # (lable, image) sorted by lable
test_lbl1 = []
test_lbl2 = []
test_lbl3 = [] # Non existing digit, say 1


test_lbl_img_list, test_lbl1, test_lbl2, test_lbl3 = get_lables('testing', first, second, third)


bayes_count = 0
test_inp = test_lbl_img_list
correct, wrong, indet = predictlable('hist', test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
print ("Hist result (Dig1 & Dig2) : correct %d wrong %d Indet %d " % (correct, wrong, indet))
correct, wrong, indet = predictlable('gaus', test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
print ("Gaus result (Dig1 & Dig2) : correct %d wrong %d Indet %d " % (correct, wrong, indet))

test_inp = test_lbl1
correct, wrong, indet = predictlable('hist', test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
print ("Hist result (Dig1): correct %d wrong %d Indet %d " % (correct, wrong, indet))
correct, wrong, indet = predictlable('gaus', test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
print ("Gaus result (Dig1): correct %d wrong %d Indet %d " % (correct, wrong, indet))

test_inp = test_lbl2
correct, wrong, indet = predictlable('hist', test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
print ("Hist result (Dig2): correct %d wrong %d Indet %d " % (correct, wrong, indet))
correct, wrong, indet = predictlable('gaus', test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
print ("Gaus result (Dig2): correct %d wrong %d Indet %d " % (correct, wrong, indet))

test_inp = test_lbl3
correct, wrong, indet = predictlable('hist', test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
print ("Hist result (OtherDigs): correct %d wrong %d Indet %d " % (correct, wrong, indet))
correct, wrong, indet = predictlable('gaus', test_inp, dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)
print ("Gaus result (OtherDigs): correct %d wrong %d Indet %d " % (correct, wrong, indet))

# Gaussian

#test_p1p2 = getp1p2oftest(test_lbl1[0][1], mean, V)

#PredictGender2dBayes(P1P2_dig1, P1P2_dig2, test_p1p2)
print "done .."
'''
correct = 0
wrong = 0
indet = 0

for lbl, vec in test_inp:
   test_p1p2 = getp1p2oftest(vec, mean, V)
   p_dig1 , p_dig2 = PredictGender2dHist(test_p1p2[0,0], test_p1p2[0,1], dig1_hist, p1_bin, p1_min, p1_max, dig2_hist, p2_bin, p2_min, p2_max)

   if p_dig1 > p_dig2:
      out = first
   elif p_dig1 < p_dig2:
      out = second
   else :
      out = -1

   if (out == lbl):
      correct = correct + 1
   elif (out == -1):
      indet = indet + 1
   else:
      wrong = wrong + 1
'''
