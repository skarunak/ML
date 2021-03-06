#!/usr/bin/python
import os
import struct
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
# Inp: List of N'd components of all digits
def BuildGender2dBayes(p_components, n_lbls):

   means = []
   covs = []
   arg1s = []
 
   rows, dimensions = p_components[0].shape
   pow_val = (float)(dimensions)/2
   print "BuildGender2dBayes : dimensions " , dimensions
   idx = 0
   
   # Use dim in the formula
   while idx < n_lbls:
     means.append(np.mean(p_components[idx], axis=0))
     covs.append(np.cov(p_components[idx], rowvar=False))
     arg1s.append((1/(((2*math.pi)**pow_val)*math.sqrt(np.linalg.det(covs[idx])))))
     idx = idx + 1

   return means, covs, arg1s 

def PredictGender2dBayes(p_comps, n_lbls, test_p_comps, means, covs, arg1s):

   p_digs = []
   p_total = 0
   probs = []
   idx = 0

   while idx < n_lbls:
      arg2 = test_p_comps - means[idx]
      #m_cov_arr = np.array(covs[idx])
      arg3 = np.linalg.inv(covs[idx]) 
      arg2_arr = np.matrix(arg2) 
      arg4 = arg2_arr.transpose()
      arg5 = np.matrix(arg2) * np.matrix(arg3)

      #print "PredictGender2dBayes: component lengths " , len(p_comps[idx])
      p_digs.append(len(p_comps[idx]) * arg1s[idx] *math.exp((-0.5)*(np.matrix(arg2) * np.matrix(arg3) * np.matrix(arg4))))
      p_total = p_total + p_digs[idx]
      idx = idx + 1

   # Probabilities ..
   idx = 0
   while idx < n_lbls:
     probs.append((float)(p_digs[idx]) / p_total)
     idx = idx + 1

   return probs

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
    digt_image_text = os.path.join(".", digit_name)
    fdesc = open (digt_image_text, "w")
    fdesc.write('P2' + "\n")
    fdesc.write(str(28) + "\n")
    fdesc.write(str(28) + "\n")
    fdesc.write(str(255) + "\n")
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

def get_lables(dataset):
    combined = []
    lbl_lengths = range(10) # Pos 0 refers to lbl1 , 1 refers to lbl2 etc
    for idx in range(10):
       lbl_lengths[idx] = 0

    for lbl, vec in read(dataset):
           #if (lbl == 4 or lbl == 3 ):
              combined.append((lbl, vec))
              lbl_lengths[lbl] = lbl_lengths[lbl]+1
    
    return combined , lbl_lengths

def getp1p2oftest(img, mean, V):
    z = img - mean
    #zVT = V[:dim,:].transpose()
    zVT = V[:,:dim]
    zp1p2 = z * zVT
    return zp1p2

# Returns list of metrics
def predictlable(classifier, test_inp, test_lbl_lengths):
   correct = 0
   wrong = 0
   indet = 0
   idx = 0
   perf = np.zeros((10,10), dtype=int)
   n_imgs = 3
   idx_08 = 0
   idx_18 = 0
   idx_23 = 0
   idx_28 = 0
   idx_38 = 0
   idx_79 = 0
   idx_60 = 0
   idx_90 = 0
   idx_0 = 0
   idx_1 = 0
   idx_2 = 0
   idx_3 = 0
   idx_7 = 0
   idx_6 = 0
   idx_9 = 0

   if classifier is 'gaus':
       mean_list, cov_list, first_arg_list = BuildGender2dBayes(P1P2_digs, len(test_lbl_lengths))

   for lbl, vec in test_inp:
     test_p1p2 = getp1p2oftest(vec, mean, V)
     if classifier is 'gaus':
       prob_digs = PredictGender2dBayes(P1P2_digs, len(test_lbl_lengths), test_p1p2, mean_list, cov_list, first_arg_list)


     max_prob = max(prob_digs)
     predicted_lbl = prob_digs.index(max_prob)

     if (predicted_lbl == lbl):
        correct = correct + 1
        if (lbl == 0):
          if idx_0 < n_imgs:
            write_image_to_text_file(vec, 'T_' + str(lbl)+ str(predicted_lbl) + '_itr_' + str(idx_0))     
            idx_0 = idx_0 + 1
        if (lbl == 1):
          if idx_1 < n_imgs:
            write_image_to_text_file(vec, 'T_' + str(lbl)+  str(predicted_lbl) + '_itr_' + str(idx_1))     
            idx_1 = idx_1 + 1
        if (lbl == 2):
          if idx_2 < n_imgs:
            write_image_to_text_file(vec, 'T_' + str(lbl)+ str(predicted_lbl) + '_itr_' + str(idx_2))     
            idx_2 = idx_2 + 1
        if (lbl == 3):
          if idx_3 < n_imgs:
            write_image_to_text_file(vec, 'T_' + str(lbl)+ str(predicted_lbl) + '_itr_' + str(idx_3))     
            idx_3 = idx_3 + 1
        if (lbl == 6):
          if idx_6 < n_imgs:
            write_image_to_text_file(vec, 'T_' + str(lbl)+ str(predicted_lbl) + '_itr_' + str(idx_6))     
            idx_6 = idx_6 + 1
        if (lbl == 7):
          if idx_7 < n_imgs:
            write_image_to_text_file(vec, 'T_' + str(lbl)+ str(predicted_lbl) + '_itr_' + str(idx_7))     
            idx_7 = idx_7 + 1
        if (lbl == 9):
          if idx_9 < n_imgs:
            write_image_to_text_file(vec, 'T_' + str(lbl)+ str(predicted_lbl) + '_itr_' + str(idx_9))     
            idx_9 = idx_9 + 1
     else:
        wrong = wrong + 1
        #print "Test lbl Prediced lbl prob " , lbl, predicted_lbl, max_prob
        #print ("Wrong:: classifier: %s idx %d " % (classifier, idx))
        #if (idx == 15):
        #   print ("Wrong:: classifier: %s idx %d " % (classifier, idx))
        #   write_image_to_text_file(vec, classifier+str(idx))     
        #if (idx == 31):
        #   print ("Wrong:: classifier: %s idx %d " % (classifier, idx))
        #   write_image_to_text_file(vec, classifier+str(idx))
        #if (idx == 393):
        #   print ("Wrong:: classifier: %s idx %d " % (classifier, idx))
        #   write_image_to_text_file(vec, classifier+str(idx))     
        #if (idx == 696):
        #   print ("Wrong:: classifier: %s idx %d " % (classifier, idx))
        #   write_image_to_text_file(vec, classifier+str(idx))
     idx = idx + 1
     perf[lbl][predicted_lbl] = perf[lbl][predicted_lbl] + 1
     if (lbl == 0 and predicted_lbl == 8):
        if idx_08 < n_imgs:
          write_image_to_text_file(vec, 'T_' + str(lbl)+ 'P_' + str(predicted_lbl) + '_itr_' + str(idx_08))     
          idx_08 = idx_08 + 1
     if (lbl == 1 and predicted_lbl == 8):
        if idx_18 < n_imgs:
          write_image_to_text_file(vec, 'T_' + str(lbl)+ 'P_' + str(predicted_lbl) + '_itr_' + str(idx_18))     
          idx_18 = idx_18 + 1
     if (lbl == 2 and predicted_lbl == 3):
        if idx_23 < n_imgs:
          write_image_to_text_file(vec, 'T_' + str(lbl)+ 'P_' + str(predicted_lbl) + '_itr_' + str(idx_23))     
          idx_23 = idx_23 + 1
     if (lbl == 2 and predicted_lbl == 8):
        if idx_28 < n_imgs:
          write_image_to_text_file(vec, 'T_' + str(lbl)+ 'P_' + str(predicted_lbl) + '_itr_' + str(idx_28))     
          idx_28 = idx_28 + 1
     if (lbl == 3 and predicted_lbl == 8):
        if idx_38 < n_imgs:
          write_image_to_text_file(vec, 'T_' + str(lbl)+ 'P_' + str(predicted_lbl) + '_itr_' + str(idx_38))     
          idx_38 = idx_38 + 1
     if (lbl == 7 and predicted_lbl == 9):
        if idx_79 < n_imgs:
          write_image_to_text_file(vec, 'T_' + str(lbl)+ 'P_' + str(predicted_lbl) + '_itr_' + str(idx_79))     
          idx_79 = idx_79 + 1
     if (lbl == 6 and predicted_lbl == 0):
        if idx_60 < n_imgs:
          write_image_to_text_file(vec, 'T_' + str(lbl)+ 'P_' + str(predicted_lbl) + '_itr_' + str(idx_60))     
          idx_60 = idx_60 + 1
     if (lbl == 9 and predicted_lbl == 0):
        if idx_90 < n_imgs:
          write_image_to_text_file(vec, 'T_' + str(lbl)+ 'P_' + str(predicted_lbl) + '_itr_' + str(idx_90))     
          idx_90 = idx_90 + 1
   
   return correct, wrong, perf

#dim = 5
lbl_img_list = [] # (lable, image) sorted by lable

lbl_img_list, lbl_lengths = get_lables('training')
lbl_img_list = sorted(lbl_img_list, key=lambda key: key[0])
print "Data set: total sz , each szs " , len(lbl_img_list), lbl_lengths

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
eg_val_list = eg_val.real.tolist()
eg_val_tot = sum(eg_val_list)

# Determine dimensions to use for 70% variance
var = []
comp = range(784)
dim = 0
for idx, val in enumerate(eg_val_list):
   agg_eg_val = sum(eg_val_list[0:idx+1])
   var.append(agg_eg_val/eg_val_tot)
   if (var[idx] >= 0.8 and dim==0): 
      dim = idx + 1

# Plot variance and principal components
#fig, ax = plt.subplots()
#ax.grid(True)
#ax.plot(comp[0:100], var[0:100])
#ax.set_xlabel("Principal components")
#ax.set_ylabel("Variance")
#plt.savefig("Var_Pcomps.png")
#plt.show()

V = V.real
V_row, V_col = V.shape
print ("V : (%d, %d) " % (V_row, V_col))

V = np.matrix(V)
#V = np.matrix(V).transpose()
#TV = V.transpose()
#P = Z * TV
P = Z * V
P_mean = np.mean(P, axis=0)
#print "P_mean ..", P_mean

P1P2 = P[:,0:dim]
P1P2_arr = np.array(P1P2)

print "P1P2 shape ", P1P2_arr.shape
P1P2_digs = range(10) # List of principal components for each digits 0 for 1, 1 for 2 etc
start = 0
end = 0
digs = 0
while digs < len(lbl_lengths):
   end = start + lbl_lengths[digs]
   P1P2_digs[digs] = P1P2_arr[start:end, 0:dim]
   start = start + lbl_lengths[digs]
   digs = digs + 1

'''
# 3d subplots
fig = plt.figure()
ax = Axes3D(fig)

plot_samples = 2500
digs = 0
colors = [np.random.rand(3,1) for i in range(10)]
proxies = []
labels = []
while digs < len(lbl_lengths):
  ax.scatter(P1P2_digs[digs][:plot_samples,0], P1P2_digs[digs][:plot_samples,1], P1P2_digs[digs][:plot_samples,2], c=colors[digs], marker='^')
  #ax.scatter(P1P2_digs[digs][:,0], P1P2_digs[digs][:,1], P1P2_digs[digs][:,2], c=colors[digs], marker='^')
  proxies.append(plt.Rectangle((0, 0), 1, 1, fc=colors[digs]))
  labels.append(digs)
  digs = digs + 1

ax.legend(proxies, labels)

ax.set_xlabel("p1")
ax.set_ylabel("p2")
ax.set_ylabel("p3")
#plt.savefig("scatter.png")
plt.show()
'''

# Recontruction with loss
VT = V.transpose()
R_ = P1P2 * VT[:dim,:]
X_ = R_ + mean

# Reconstruction images
idx = 0
while idx < len(lbl_lengths): 
   write_image_to_text_file(np.array(X_[sum(lbl_lengths[:idx])]), 'rec_1_'+  str(lbl_img_list[sum(lbl_lengths[:idx])][0]))
   write_image_to_text_file(np.array(X_[sum(lbl_lengths[:idx])+1]), 'rec_2_'+ str(lbl_img_list[sum(lbl_lengths[:idx])+1][0]))
   write_image_to_text_file(np.array(X_[sum(lbl_lengths[:idx])+2]), 'rec_3_'+str(lbl_img_list[sum(lbl_lengths[:idx])+2][0]))
   write_image_to_text_file(np.array(X_arr[sum(lbl_lengths[:idx])]), 'orig_1_'+  str(lbl_img_list[sum(lbl_lengths[:idx])][0]))
   write_image_to_text_file(np.array(X_arr[sum(lbl_lengths[:idx])+1]), 'orig_2_'+ str(lbl_img_list[sum(lbl_lengths[:idx])+1][0]))
   write_image_to_text_file(np.array(X_arr[sum(lbl_lengths[:idx])+2]), 'orig_3_'+str(lbl_img_list[sum(lbl_lengths[:idx])+2][0]))
   idx = idx + 1

# Loss less reconstruction
#R = P * V

#test_lbl_img_list = [] # (lable, image) sorted by lable
test_lbl_img_list, test_lbl_lengths = get_lables('testing')

bayes_count = 0
test_inp = test_lbl_img_list

correct, wrong, perf = predictlable('gaus', test_lbl_img_list, test_lbl_lengths)
print ("Gaus result (AllDigs) : correct %d wrong %d " % (correct, wrong))
print "Performance .. \n" , perf

print "done .."

