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
    digt_image_text = os.path.join(".", digit_name+'_image_text')
    fdesc = open (digt_image_text, "w")
    for count in range (len(image_array)):
        fdesc.write(str(image_array[count]) + "\n")
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

#Filter 3 & 4

lbl_img_list = [] # (lable, image) sorted by lable
lbl1 = []
lbl2 = []

for lbl,vec in read():
   if (lbl == 3 or lbl == 4):
      lbl_img_list.append((lbl, vec))
      if (lbl == 3):
         lbl1.append((lbl, vec))
      if (lbl == 4):
         lbl2.append((lbl, vec))

lbl_img_list = sorted(lbl_img_list, key=lambda key: key[0])
print ("Data set: sz %d " % (len(lbl_img_list)))

X_arr = np.stack((j for i,j in lbl_img_list))

print "X array " , X_arr[0]
mean = np.mean(X_arr, axis=0)
print "mean array ", mean

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
print ("Eig.Val : (%d, %d) " % (eg_val[0].real, eg_val[1].real))
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

print "done .."
'''
# Print to file 
print "Label : ", lbl_img_list[1200][0]
for row in lbl_img_list[1200][1]:
   for ch in row:
      print ch
'''
