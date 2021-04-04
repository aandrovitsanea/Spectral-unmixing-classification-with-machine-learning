#!/usr/bin/env python
# coding: utf-8

# **<span style="font-family:KerkisSans; font-size:2.5em;">Processing in Hyperspectral Images (HSIs)</span>**
# 
# * **<span style="font-family:KerkisSans; font-size:2.5em; color: red">Spectral unmixing</span>**
# 
# 
# * **<span style="font-family:KerkisSans; font-size:2.5em; color: red">Classification (supervised, unsupervised)</span>**
# 
# <span style="font-family:KerkisSans; font-size:1.5em; color: black">Anna Androvitsanea</span>
# 
# <span style="font-family:KerkisSans; font-size:1.5em; color: black">anna.androvitsanea@gmail.com</span>

# **<span style="font-family:KerkisSans; font-size:2.5em;">Table of contents</span>**
# 
# * [Introduction](#Introduction)
#     * [Import libraries](#Import-libraries)
#     * [Import data](#Import-data-for-regression)
#     * [Plot data](#Plots)
#         * [Spectral signatures for 9 endmembers](#Spectral-signatures-for-9-endmembers)
#         * [Spectral signatures for 9 endmembers](#section_1_2_2)
#         * [Ground truth](#Ground-truth)
#         * [Ground truth masked](#Ground-truth-masked)
#         * [RGB Visualization of the 10th band](#RGB-Visualization-of-the-10th-band)
# * [Part 1: Spectral unmixing (SU)](#Part-1:-Spectral-unmixing-(SU))
#     * [Scope](#Scope)
#     * [(a) Least squares](#(a)-Least-squares)
#     * [(b) Least squares imposing the sum-to-one constraint](#(b)-Least-squares-imposing-the-sum-to-one-constraint)
#     * [(c) Least squares imposing the non-negativity constraint](#(c)-Least-squares-imposing-the-non-negativity-constraint)
#     * [(d) Least squares imposing both the sum-to-one and the non-negativity constraint](#(d)-Least-squares-imposing-both-the-non-negativity-and-the-sum-to-one-constraint)
#     * [e) LASSO, impose sparsity via l1 norm minimization](#(e)-LASSO,-impose-sparsity-via-l_1-norm-minimization)
#     * [Comparison of regressors](#Comparison-of-regressors)
# * [Part 2: Spectral unmixing (SU)](#Part-1:-Spectral-unmixing-(SU))
#     * [Import data for classification](#Import-data-for-classification)
#     * [Prepare data for classification](#Prepare-data-for-classification)
#     * [Plots](#Plots)
#     * [Scope](#Scopes)
#     * [(A) Classification for each classifier](#(A)-Classification-for-each-classifier)
#         * [Naive Bayes classifier](#Naive-Bayes-classifier)
#         * [Minimum Euclidean distance classifier](#Minimum-Euclidean-distance-classifier)
#         * [k-nearest neighbor classifier](#k-nearest-neighbor-classifier)
#         * [Bayesian classifier](#Bayesian-classifier)
#     * [(B) Comparison of classifiers](#(B)-Comparison-of-classifiers)
# * [Part 3: Combination - Correlation](#Part-3:-Combination---Correlation)

# # Introduction

# ## Import libraries

# In[1]:


# import libraries

import scipy.io as sio
import pandas as pd
import numpy as np
import scipy.optimize 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.colors as clr
from numpy import linalg as LA
from scipy.optimize import nnls 
from cvxopt import matrix, solvers
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# ## Import data for regression

# In[2]:


# import data 

Pavia = sio.loadmat('PaviaU_cube.mat')
HSI = Pavia['X'] #Pavia HSI : 300x200x103

ends = sio.loadmat('PaviaU_endmembers.mat') # Endmember's matrix: 103x9
endmembers = ends['endmembers']


# In[3]:


# import ground truth

ground_truth = sio.loadmat('PaviaU_ground_truth.mat')
labels = ground_truth['y']


# ## Plots

# ### Spectral signatures for 9 endmembers

# In[4]:


# plot spectral signatures for 9 endmembers

fig = plt.figure(figsize=(10,4))
plt.plot(endmembers)
plt.ylabel('Radiance values')
plt.xlabel('Spectral bands')
plt.title('9 Endmembers spectral signatures at Pavia University HSI')
plt.show()


# ### Spectral signatures for each endmember

# In[5]:


# make a dict mapping the 9 endmembers to the corresponding material name

materials = {0: 'unknown',
             1: 'Water',
             2: 'Trees',
             3: 'Asphalt',
             4: 'Bricks',
             5: 'Bitumen',
             6: 'Tiles',
             7: 'Shadows',
             8: 'Meadows',
             9: 'Bare Soil'}


# In[6]:


# Plot the spectral signature for each endmember


fig = plt.figure()
colors = ['royalblue','green','dimgray','firebrick', 'silver',
          'cadetblue','black','lawngreen', 'chocolate']
for i in range (0,9):
    plt.plot(endmembers[:,0], c=colors[i])
    plt.ylabel('Radiance values')
    plt.xlabel('Spectral bands')
    plt.title('Spectral signature of endmember %d (%s) of Pavia University HSI' % (i+1, materials[i+1]))
    plt.show()


# ### Ground truth

# In[7]:


# plot labels, icluding zero values

plt.figure(figsize=(20,10)) # set fig size

# make a dict to assign a distinct color to each material/surface
color_dict= {0: 'oldlace', 1: 'royalblue', 2: 'green', 3: 'dimgray', 4: 'firebrick', 5: 'silver',
             6: 'cadetblue', 7: 'black', 8: 'lawngreen', 9: 'chocolate'}
cmaps_shuffle = {1:'Blues',2:'Greens',3:'Greys', 4:'Oranges', 5:'gist_yarg', 
               6:'Purples',7:'Greys',8:'Greens',9:'YlOrBr'}

# make a color map form the dict
cmap = ListedColormap([color_dict[x] for x in color_dict.keys()])

# make a legend based on the color and material
patches = [mpatches.Patch(color=color_dict[i],label = materials[i]) for i in range(0,10)]

plt.legend(handles=patches, bbox_to_anchor = (1.3, 1), loc=2, borderaxespad=0. )

plt.imshow(labels, cmap = cmap) # plot the array

plt.title('Visualization of the ground truth for the land cover\nof Pavia University HSI')

plt.colorbar() # plot the color bar

plt.savefig('ground_truth_raw.png') # save figure


# ### Ground truth masked

# In[8]:


# plot labels without the zero values

X = np.ma.masked_equal(labels, 0) # mask zero values 


plt.figure(figsize=(20,10)) # set fig size

# make a dict to assign a distinct color to each material/surface
color_dict_masked = {1: 'royalblue', 2: 'green', 3: 'dimgray', 4: 'firebrick', 5: 'silver',
                     6: 'cadetblue', 7: 'black', 8: 'lawngreen', 9: 'chocolate'}

materials_masked = {1: 'Water',
                    2: 'Trees',
                    3: 'Asphalt',
                    4: 'Bricks',
                    5: 'Bitumen',
                    6: 'Tiles',
                    7: 'Shadows',
                    8: 'Meadows',
                    9: 'Bare Soil'}

# make a color map form the dict
cmap_masked = ListedColormap([color_dict_masked[x] for x in color_dict_masked.keys()])

# make a legend based on the color and material
patches_masked =[mpatches.Patch(color=color_dict_masked[i],label=materials_masked[i]) for i in range(1,10)]

plt.legend(handles=patches_masked, bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=0. )

plt.imshow(X, cmap = cmap_masked) # plot the array

plt.title('Visualization of the ground truth with masked zero values \nfor the land cover of Pavia University HSI')

plt.colorbar() # plot the color bar

plt.savefig('ground_truth_raw_masked.png') # save figure


# ### RGB Visualization of the 10th band

# In[9]:


# plot data
plt.figure(figsize=(20,10))


plt.imshow(HSI[:,:,10], cmap = 'inferno')
plt.title('RGB Visualization of the $10^{th}$ band of Pavia University HSI')
plt.show()


# # Part 1: Spectral unmixing (SU)

# ## Scope

# In this part I consider the set consisting of the 9 endmembers, where each endmember has a specific spectral signature, which corresponds to the pure pixels in the HSI dataset. 
# 
# For a given pixel in the image, the aim is to determine the percentage (abundance) that each pure material contributes in its formation and **unmix** the pixels.
# 
# I fit linear regression models that connect the 9 endmembers with the mixed pixels of the HSI dataset.
# I consider the following models:
# 
# (a) [Least squares](#(a)-Least-squares)
# 
# (b) [Least squares imposing the sum-to-one constraint for $\theta$s,](#(b)-Least-squares-imposing-the-sum-to-one-constraint)
# 
# (c) [Least squares imposing the sum-to-one constraint for $\theta$s,](#(c)-Least-squares-imposing-the-non-negativity-constraint)
# 
# (d) [Least squares imposing both the non-negativity and the sum-to-one constraint for $\theta$s,](#(d)-Least-squares-imposing-both-the-non-negativity-and-the-sum-to-one-constraint)
# 
# (e) [LASSO, imposing sparsity on $\theta$s via $l_1$ norm minimization.](#(e)-LASSO-impose-sparsity-via-l_1-norm-minimization)
# 
# First, I calculate the abundance maps for each material, i.e. 9 maps. 
# 
# Then I compute the reconstruction error as follows:
# 
# I calculate the reconstruction error (for each non-zero class label) of **each pixel** using the formula:
#     $$error = \frac{||\mathbf{y}_i - \mathbf{X}\mathbb{\theta}_i||^2}{||\mathbf{y}_i||^2}$$
# Then, for **N pixels** I compute the **average** value:
#     $$\text{reconstruction error} = \frac{error}{N} $$
#     
# Finaly, I [compare](#Comparison-of-regressors) the results obtained from the above five methods based on the abundance maps and the reconstruction error.
#     
#     

# ## (a) Least squares

# In[10]:


# Linear regression without constraints
# I perform the regression for those pixels
# that have non zero label
# at the ground truth matrix
# Those with zero label will get a zero array
# I want to ensure the integrity
# of the rows x columns of the image

XTXinv = np.linalg.inv(endmembers.T @ endmembers) # calculate the endmembers.T * endmembers

thetas = []
for row in range (0, 300):
    for column in range (0, 200):
        if labels[row][column] != 0: # perform the regression for the non zero values
            y = np.reshape(HSI[row,column,:], (103, 1)) # reshape the mixed signature of the pixel
            theta = XTXinv @ endmembers.T @ y # calculate the theta estimators
        else:
            theta = np.reshape(np.zeros(9), (9,1)) # store a 9x1 array with zeros
        thetas.append(theta)
                
        
thetas_ar = np.array(thetas) # transform to array
thetas_ar_resh = np.reshape(thetas_ar, (300,200, 9))
thetas_ar_resh.shape


# In[11]:


# unmixing the pixel

y_unmixed_ols = thetas_ar_resh @ endmembers.T # calculate unmixed pixel with reshaped array
y_unmixed_ols.shape


# ### Plot: $\theta$ vs masked ground truth

# In[12]:


# plot thetas for all endmembers vs the masked ground truth


for endmember in range(0,9):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    
    im1 = ax1.imshow(thetas_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember + 1])
    im2 = ax2.imshow(np.ma.masked_where(labels != (endmember + 1), 
                                        labels), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    fig.colorbar(im2, cax=cax2, orientation='vertical')
    
    plt.subplots_adjust(right=1.0)
    ax1.set_title(r'Abundance map: $\theta_{%d}$: Endmember %d' % 
                 (endmember+1, endmember+1))
    ax2.set_title('Ground truth')
    ax2.legend(handles=patches_masked, 
               bbox_to_anchor=(1.3, 1), 
               loc=2, borderaxespad=0.)
    fig.suptitle(r'Linear regression: $\theta_{%d}$ value vs ground truth' % 
                 (endmember+1))
    
    


# ### Reconstruction error

# In[13]:


# calculate the reconstruction error

reco_error = 0
N = 0

for row in range (0, 300):
    for column in range (0, 200):
        if labels[row][column] != 0: # perform the regression for the non zero values
            y = np.reshape(HSI[row,column,:], (103, 1)) # reshape the mixed signature of the pixel
            theta_reshaped = np.reshape(thetas_ar_resh[row,column,:], (9,1)) # reshape the thetas matrix
            
            error_init = ((LA.norm(y - endmembers @ theta_reshaped))**2) / ((LA.norm(y))**2) # error for pixel
            N += 1 # keep count of calculated pixels
            
            reco_error += error_init # sum of errors

reconstruction_error_ls = reco_error / N # average value of all pixels' reconstruction errors
print('The reconstruction error is:', reconstruction_error_ls)


# ## (b) Least squares imposing the sum-to-one constraint

# I am going to apply the ```cvxopt``` function in order to introduce constraints to the system of linear regression.
# 
# Quadratic systems can be solved via the ```solvers.qp()``` function. As an example, I can solve the quadratic problem
# 
# \begin{array}{ll} \mbox{minimize} & (1/2) \theta^TP\theta + q^T x \\ \mbox{subject to} & G * \theta \preceq h \\ & A\theta = b \end{array}
# 
# In this case I want to minimize the problem:
# 
# \begin{array}{ll} ||\mathbf{y} - \mathbf{x}\mathbf{\theta} ||^2 \Rightarrow \\ (\mathbf{y} - \mathbf{x}\mathbf{\theta})^T(\mathbf{y} - \mathbf{x}\mathbf{\theta}) \Rightarrow \\ (\mathbf{y}^T - \mathbf{x}^T\mathbf{\theta}^T) (\mathbf{y} - \mathbf{x}\mathbf{\theta}) \Rightarrow \\  \mathbf{y}^T\mathbf{y} - \mathbf{x}^T\mathbf{\theta}^T\mathbf{y}  - \mathbf{y}^Τ\mathbf{x}\mathbf{\theta} + \mathbf{x}^Τ\mathbf{\theta}^Τ \mathbf{x}\mathbf{\theta}\\ \mbox{I ignore the element } \mathbf{y}^T\mathbf{y} \mbox{ since is not subjct to $\theta$}: \boxed{\mathbf{\theta}^T\mathbf{x}^T\mathbf{x}\mathbf{\theta}-2\mathbf{y}^T \mathbf{x} \mathbf{\theta} } \Rightarrow \\ P = 2\mathbf{x}^T\mathbf{x} \mbox{ and } q^T = -2\mathbf{y}^T\mathbf{x} \end{array}
# 
# * Since I want to impose the sum-to-one constraint, I set $\mathbf{A}$ = $\mathbf{b}$ = $\mathbf{1}$

# In[14]:


thetas_sum_one = []

A = matrix(np.ones((1,9)))

b = matrix(np.array([1.]))

P = matrix(2 * endmembers.T @ endmembers)

for row in range (0, 300):
    for column in range (0, 200):
        if labels[row][column] != 0: # perform the regression for the non zero values
            q = -2 * matrix(endmembers.T @ HSI[row,column,:])
            sol = solvers.qp(P, q, A = A, b = b)
            theta = np.array(sol['x'])
        else:
            theta = np.reshape(np.zeros(9), (9,1)) # store a 9x1 array with zeros
        thetas_sum_one.append(theta)
                
        
thetas_sum_one_ar = np.array(thetas_sum_one) # transform to array
thetas_sum_one_ar_resh = np.reshape(thetas_sum_one_ar, (300,200, 9))
thetas_sum_one_ar_resh.shape


# In[15]:


# unmixing the pixel

y_unmixed_sum_one = thetas_sum_one_ar_resh @ endmembers.T # calculate unmixed pixel with reshaped array
y_unmixed_sum_one.shape


# ### Plot: $\theta$ vs masked ground truth

# In[16]:


# plot thetas for all endmembers vs the masked ground truth


for endmember in range(0,9):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    
    im1 = ax1.imshow(thetas_sum_one_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember + 1])
    im2 = ax2.imshow(np.ma.masked_where(labels != (endmember + 1), 
                                        labels), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    fig.colorbar(im2, cax=cax2, orientation='vertical')
    
    plt.subplots_adjust(right=1.0)
    ax1.set_title(r'Abundance map: $\theta_{%d}$: Endmember %d' % 
                 (endmember+1, endmember+1))
    ax2.set_title('Ground truth')
    ax2.legend(handles=patches_masked, 
               bbox_to_anchor=(1.3, 1), 
               loc=2, borderaxespad=0.)
    fig.suptitle(r'Linear regression under sum-to-one constraint: $\theta_{%d}$ value vs ground truth' % 
                 (endmember+1))
    
    


# ### Reconstruction error ls sum-one

# In[17]:


# calculate the reconstruction error

reco_error = 0
N = 0

for row in range (0, 300):
    for column in range (0, 200):
        if labels[row][column] != 0: # perform the regression for the non zero values
            y = np.reshape(HSI[row,column,:], (103, 1)) # reshape the mixed signature of the pixel
            theta_reshaped = np.reshape(thetas_sum_one_ar_resh[row,column,:], (9,1)) # reshape the thetas matrix
            
            error_init = ((LA.norm(y - endmembers @ theta_reshaped))**2) / ((LA.norm(y))**2) # error for pixel
            N += 1 # keep count of calculated pixels
            
            reco_error += error_init # sum of errors

reconstruction_error_sum_one = reco_error / N # average value of all pixels' reconstruction errors
print('The reconstruction error is:', reconstruction_error_sum_one)


# ## (c) Least squares imposing the non-negativity constraint

# Quadratic programs can be solved via the ```solvers.qp()``` function [as above](#(b)-Least-squares-imposing-the-sum-to-one-constraint) .
# 
# * Since I want to impose the non-negativity constraint, I set $\mathbf{G} = \mathbf{-1}$ and $\mathbf{h} = \mathbf{0}$

# In[18]:


thetas_non_neg = []

P = matrix(2 * endmembers.T @ endmembers)

G = - matrix(np.eye(9))

h = matrix(np.zeros(9))

for row in range (0, 300):
    for column in range (0, 200):
        if labels[row][column] != 0: # perform the regression for the non zero values
            q = -2 * matrix(endmembers.T @ HSI[row,column,:])
            sol = solvers.qp(P, q, G = G, h = h, options = {'show_progress': False})
            theta = np.array(sol['x'])
        else:
            theta = np.reshape(np.zeros(9), (9,1)) # store a 9x1 array with zeros
        thetas_non_neg.append(theta)
                
        
thetas_non_neg_ar = np.array(thetas_non_neg) # transform to array
thetas_non_neg_ar_resh = np.reshape(thetas_non_neg_ar, (300,200, 9))
thetas_non_neg_ar_resh.shape


# In[19]:


# unmixing the pixel

y_unmixed_non_neg = thetas_non_neg_ar_resh @ endmembers.T # calculate unmixed pixel with reshaped array
y_unmixed_non_neg.shape


# ### Plot: $\theta$ vs masked ground truth

# In[20]:


for endmember in range(0,9):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    
    im1 = ax1.imshow(thetas_non_neg_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember + 1])
    im2 = ax2.imshow(np.ma.masked_where(labels != (endmember + 1), 
                                        labels), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    fig.colorbar(im2, cax=cax2, orientation='vertical')
    
    plt.subplots_adjust(right=1.0)
    ax1.set_title(r'Abundance map: $\theta_{%d}$: Endmember %d' % 
                 (endmember+1, endmember+1))
    ax2.set_title('Ground truth')
    ax2.legend(handles=patches_masked, 
               bbox_to_anchor=(1.3, 1), 
               loc=2, borderaxespad=0.)
    fig.suptitle(r'Linear regression under non-negativity constraint: $\theta_{%d}$ value vs ground truth' % 
                 (endmember+1))


# ### Reconstruction error ls non-neg

# In[21]:


# calculate the reconstruction error

reco_error = 0
N = 0

for row in range (0, 300):
    for column in range (0, 200):
        if labels[row][column] != 0: # perform the regression for the non zero values
            y = np.reshape(HSI[row,column,:], (103, 1)) # reshape the mixed signature of the pixel
            theta_reshaped = np.reshape(thetas_non_neg_ar_resh[row,column,:], (9,1)) # reshape the thetas matrix
            
            error_init = ((LA.norm(y - endmembers @ theta_reshaped))**2) / ((LA.norm(y))**2) # error for pixel
            N += 1 # keep count of calculated pixels
            
            reco_error += error_init # sum of errors

reconstruction_error_non_neg = reco_error / N # average value of all pixels' reconstruction errors
print('The reconstruction error is:', reconstruction_error_non_neg)


# ## (d) Least squares imposing both the non-negativity and the sum-to-one constraint

# Quadratic programs can be solved via the ```solvers.qp()``` function [as above](#(b)-Least-squares-imposing-the-sum-to-one-constraint) .
# 
# * Since I want to impose the sum-to-one constraint, I set $\mathbf{A} = \mathbf{b} = \mathbf{1}$
# * Since I want to impose the non-negativity constraint, I set $\mathbf{G} = \mathbf{-1}$ and $\mathbf{h} = \mathbf{0}$

# In[22]:


thetas_sum_one_non_neg = []

A = matrix(np.ones((1,9)))

b = matrix(np.array([1.]))

P = matrix(2 * endmembers.T @ endmembers)

G = -matrix(np.eye(9))

h = matrix(np.zeros(9))

for row in range (0, 300):
    for column in range (0, 200):
        if labels[row][column] != 0: # perform the regression for the non zero values
            q = -2 * matrix(endmembers.T @ HSI[row,column,:])
            sol = solvers.qp(P, q, A = A, b = b, G = G, h = h, options = {'show_progress': False})
            theta = np.array(sol['x'])
        else:
            theta = np.reshape(np.zeros(9), (9,1)) # store a 9x1 array with zeros
        thetas_sum_one_non_neg.append(theta)
                
        
thetas_sum_one_non_neg_ar = np.array(thetas_sum_one_non_neg) # transform to array
thetas_sum_one_non_neg_ar_resh = np.reshape(thetas_sum_one_non_neg_ar, (300,200, 9))
thetas_sum_one_non_neg_ar_resh.shape


# In[23]:


# unmixing the pixel
# calculate unmixed pixel with reshaped array
y_unmixed_sum_one_non_neg = thetas_sum_one_non_neg_ar_resh @ endmembers.T 
y_unmixed_sum_one_non_neg.shape


# ### Plot: $\theta$ vs masked ground truth

# In[24]:


# plot thetas for all endmembers vs the masked ground truth


for endmember in range(0,9):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    
    im1 = ax1.imshow(thetas_sum_one_non_neg_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember + 1])
    im2 = ax2.imshow(np.ma.masked_where(labels != (endmember + 1), 
                                        labels), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    fig.colorbar(im2, cax=cax2, orientation='vertical')
    
    plt.subplots_adjust(right=1.0)
    ax1.set_title(r'Abundance map: $\theta_{%d}$: Endmember %d' % 
                 (endmember+1, endmember+1))
    ax2.set_title('Ground truth')
    ax2.legend(handles=patches_masked, 
               bbox_to_anchor=(1.3, 1), 
               loc=2, borderaxespad=0.)
    fig.suptitle(r'Linear regression under non-negativity and the sum-to-one constraint: $\theta_{%d}$ value vs ground truth' % 
                 (endmember+1))


# ### Reconstruction error ls all-const

# In[25]:


# calculate the reconstruction error

reco_error = 0
N = 0

for row in range (0, 300):
    for column in range (0, 200):
        if labels[row][column] != 0: # perform the regression for the non zero values
            y = np.reshape(HSI[row,column,:], (103, 1)) # reshape the mixed signature of the pixel
            theta_reshaped = np.reshape(thetas_sum_one_non_neg_ar_resh[row,column,:], (9,1)) # reshape the thetas matrix
            
            error_init = ((LA.norm(y - endmembers @ theta_reshaped))**2) / ((LA.norm(y))**2) # error for pixel
            N += 1 # keep count of calculated pixels
            
            reco_error += error_init # sum of errors

reconstruction_error_non_neg_sum_one = reco_error / N # average value of all pixels' reconstruction errors
print('The reconstruction error is:', reconstruction_error_non_neg_sum_one)


# ## (e) LASSO, impose sparsity via l_1 norm minimization

# I want to minimize the quantity $(1 / (2 * n_{samples})) * ||Y - X_W||^2_{Fro} + \alpha * ||W||_211$ by imposing sparcity on $\theta$ with  L1/L2 mixed-norm as regularizer.
# 
# $||W||_{21} = \sum_i \sqrt{\sum_j w_{ij}^2}$ is the sum of norm of each row.
# 
# I will implement the class ```sklearn.linear_model.MultiTaskLasso()``` which calculates the Lasso linear model with iterative fitting along a regularization path.

# In[26]:


# set the model and fit the non zero HSI values

no_zero = np.where(labels!=0)

HSI_no_zero = HSI[no_zero[0],no_zero[1],:].T

lasso_model = MultiTaskLasso(alpha = 1e+6, tol = 0.01,
                                 max_iter = 1e+4, warm_start = True,
                                 fit_intercept = False)
fitted_lasso = lasso_model.fit(endmembers, HSI_no_zero)
fitted_lasso.coef_.shape


# In[27]:


# transform the array of thetas from LASSO 
# to a 300 x 200 x 9 array (image)

lasso_map = np.zeros((300,200,9))
for index in range(len(no_zero[0])):
    row, col = no_zero[0][index],no_zero[1][index]
    for endmember in range(0, 9):
        lasso_map[row, col, endmember] = fitted_lasso.coef_[index, 
                                                            endmember]
        
lasso_abund_map = np.ma.masked_equal(lasso_map, 0) # mask zero values 


# In[28]:


# unmixing the pixel

unmixed_lasso = endmembers @ fitted_lasso.coef_.T
unmixed_lasso.T.shape

# arrange in 300 * 200 array

unmixed_lasso_array = np.zeros((300,200,9))
for index in range(len(no_zero[0])):
    row, col = no_zero[0][index],no_zero[1][index]
    for endmember in range(0, 9):
        unmixed_lasso_array[row, col, endmember] = unmixed_lasso.T[index, 
                                                            endmember] 


# ### Plot: $\theta$ vs masked ground truth

# In[29]:


# plot thetas for all endmembers vs the masked ground truth


for endmember in range(0,9):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    
    im1 = ax1.imshow(lasso_abund_map[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember + 1])
    im2 = ax2.imshow(np.ma.masked_where(labels != (endmember + 1), 
                                        labels), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    fig.colorbar(im2, cax=cax2, orientation='vertical')
    
    plt.subplots_adjust(right=1.0)
    ax1.set_title(r'Abundance map: $\theta_{%d}$: Endmember %d' % 
                 (endmember+1, endmember+1))
    ax2.set_title('Ground truth')
    ax2.legend(handles=patches_masked, 
               bbox_to_anchor=(1.3, 1), 
               loc=2, borderaxespad=0.)
    fig.suptitle(r'Linear regression under LASSO constraint: $\theta_{%d}$ value vs ground truth' % 
                 (endmember+1))


# ### Reconstruction error OLS LASSO

# In[30]:


# find the reconstruction error for lasso

def errors_lasso(matrix, no_zero):
    error = np.zeros((300,200))
    unmixed = endmembers @ matrix
    for i in range(len(no_zero[0])):
        row, col = no_zero[0][i],no_zero[1][i]
        y_min_x_theta = np.linalg.norm(HSI[row, col,: ] -
                                       unmixed[:, i]) **2
        y_sq = np.linalg.norm(HSI[row, col, :]) **2
        error[row, col] = y_min_x_theta / y_sq
    return error

lasso_errors = errors_lasso(fitted_lasso.coef_.T, np.where(labels != 0))
reco_lasso_error = lasso_errors.sum() / len(np.where(labels!=0)[0])

print('The reconstruction error is:', reco_lasso_error)


# ## Comparison of regressors

# ### Reconstruction error

# In[31]:


# Least squares errors
print('Least squares error = {:>30}' .format(round(reconstruction_error_ls,4)))
print('Least squares sum-to-one error = {:>18}'.format(round(reconstruction_error_sum_one,4)))
print('Least squares non-negative error = {:>17}'.format(round(reconstruction_error_non_neg,4)))
print('Least squares sum-to-one non-negative error = {:>}'.format(round(reconstruction_error_non_neg_sum_one,4)))
print('Least squares LASSO error = {:>24}' .format(round(reco_lasso_error,4)))


# Among the regressors I examine, the **minimum** least squares error is achieved by the **ols** regressor and the **maximum** error by  **ols** with the **sum-to-one and non-negative** constraint for the $\theta_i$ values.
# 
# However, I want to see how the regressors perform spatially, i.e. how they reconstruct the map of the campus.
# 
# For this scope, I will test each material separately.

# ### Plots

# In[32]:


# set a function to plot each label in order to compare on a
# one-by-one basis

def function_to_plot(endmember):

    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(figsize = (30,20), nrows = 2, ncols = 3)

    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    divider3 = make_axes_locatable(ax3)
    divider4 = make_axes_locatable(ax4)    
    divider5 = make_axes_locatable(ax5)    
    divider6 = make_axes_locatable(ax6)
    
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    cax6 = divider6.append_axes('right', size='5%', pad=0.05)
    
    im1 = ax1.imshow(thetas_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember+1])
    im2 = ax2.imshow(thetas_sum_one_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember+1]) 
    im3 = ax3.imshow(thetas_non_neg_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember+1])    
    im4 = ax4.imshow(thetas_sum_one_non_neg_ar_resh[:,:,endmember], 
                  cmap = cmaps_shuffle[endmember+1]) 
    im5 = ax5.imshow(lasso_abund_map[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember+1])  
    
    im6 = ax6.imshow(np.ma.masked_where(labels != (endmember + 1), 
                                        labels), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    fig.colorbar(im2, cax=cax2, orientation='vertical')
    fig.colorbar(im3, cax=cax3, orientation='vertical')    
    fig.colorbar(im4, cax=cax4, orientation='vertical')
    fig.colorbar(im5, cax=cax5, orientation='vertical')    
    fig.colorbar(im6, cax=cax6, orientation='vertical')   
      
    ax1.set_title(r'Linear Reg.: No constraint', 
                  fontsize = 25)
    ax2.set_title('Sum-to-one', fontsize = 25)
    ax3.set_title('Non-negativity', fontsize = 25)
    ax4.set_title('Sum-to-one\n non-negativity', fontsize = 25)
    ax5.set_title('LASSO', fontsize = 25)   
    ax6.set_title('Ground truth for endmember %d' % 
                  ((endmember+1)), fontsize = 25)
    fig.suptitle(r'Abundance maps of the linear regressors: $\theta_{%d}$ value vs ground truth for %s' % 
                 ((endmember+1), materials_masked[endmember+1]), fontsize = 30)  
    


# #### Water

# In[33]:


function_to_plot(0)


# Comparing the ground truth for water against the abundance maps of the regressors I notice the following:
# * The **non-negativity** constraint regressor succeeds in representing the torrent that crosses the campus diagonally, while it only assigns a similar $\theta$ value to very few other landscape features. 
# * The regressor with the **sum-to-one and non-negativity** constraints comes in second. It also identifies the torrent and only assigns a similar $\theta$ value to few other landscape features. 
# * The simple ols and the sum-to-one methods identify the torrent but assign incorrectly a similar $\theta$ value to a lot of other landscape features.
# * Finally, the LASSO regressor scores averagely. It identifies the torrent but incorrectly assigns a similar $\theta$ value to many other landscape features.

# #### Trees

# In[34]:


function_to_plot(1)


# Comparing the ground truth for trees against the abundance maps of the regressors I notice the following:
# * The **non-negativity** constraint regressor succeeds in identifying the park of trees at the north of the campus, but assigns a similar $\theta$ value to the roof of a building. 
# * The second best performance is by the regressor with the **sum-to-one and non-negativity** constraint. It also identifies the trees, while incorrectly assigning a similar $\theta$ value to few other landscape features. 
# * The simple ols and the sum-to-one methods identify the trees but assign incorrectly a similar $\theta$ value to numerous other landscape features.
# * Finally, the LASSO regressor is in the middle, identifying the trees but incorrectly assigning a similar $\theta$ value to many other landscape features.

# #### Asphalt

# In[35]:


function_to_plot(2)


# Comparing the ground truth for asphalt against the abundance maps of the regressors I notice the following:
# * The **sum-to-one and non-negativity** constraint regressors succeeds in identifying the roads at the campus, while assigning a similar $\theta$ value to some other landscape features. 
# * The second best performance is of the regressor with the **non-negativity** constraints. It also manages to identify the roads, but incorrectly assigns a similar $\theta$ value to some more landscape features. 
# * The simple ols and the sum-to-one methods identify the roads but incorrectly assign a similar $\theta$ value to a lot of other landscape features.
# * Finally, the LASSO regressor scores in the middle, identifying the roads but incorrectly asigning a similar $\theta$ value to many other landscape features.
# 
# **<span style="font-size:1.2em; color: black">None of the five regression models produce a satisfactory result
#     when identifying the roads (asphalt).</span>**
# 

# #### Bricks

# In[36]:


function_to_plot(3)


# Comparing the ground truth for bricks against the abundance maps of the regressors I notice the following:
# 
# * The **sum-to-one and non-negativity** and **non-negativity** constraint regressors both succeed in identifying the bricks at the campus, while assigning a similar $\theta$ value to a few other landscape features. 
# * The regressor with the **non-negativity** constraint comes in third. It identifies the bricks, but incorrectly assigns a similar $\theta$ value to a larger number of other landscape features. 
# * The simple ols and the sum-to-one methods identify the bricks but incorrectly assign a similar $\theta$ value to a lot of other landscape features.
# * Finally, LASSO regressor is only moderately successful, identifying the bricks but also incorrectly assigning a similar $\theta$ value to many other landscape features.
# 
# **<span style="font-size:1.2em; color: black">None of the five regression models produce a satisfactory result when identifying the bricks.</span>**
# 

# #### Bitumen

# In[37]:


function_to_plot(4)


# Comparing the ground truth for bitumen against the abundance maps of the regressors I notice the following:
# 
# * The **non-negativity** constraint regressor succeeds almost **perfectly** in identifying the bitumen at the campus.
# 
# * The **simple ols** and **sum-to-one** constraint regressors are close behind; they also identify the bitumen, while incorrectly assigning a similar $\theta$ value to some other landscape features. 
# 
# * The sum-to-one non-negativity constraint regressor identifies the bitumen but incorrectly assigns a similar $\theta$ value to a lot of other landscape features.
# 
# * Finally, the LASSO regressor scores very poorly. It identifies the bitumen but incorrectly assigns a similar $\theta$ value to many other landscape  features.
# 
# 

# #### Tiles

# In[38]:


function_to_plot(5)


# Comparing the ground truth for tiles against the abundance maps of the regressors I notice the following:
# 
# * The **non-negativity** constraint regressor succeeds almost **perfectly** in identifying the tiles at the campus.
# 
# * The second best performance is that of the **sum-to-one non-negativity** constraint regressor which also identifies the tiles, while incorrectly assigning a similar $\theta$ value to the trees. 
# 
# * The simple ols and the sum-to-one constraint regressors correctly identify the tiles but they incorrectly assign a similar $\theta$ value to a lot of other landscape features.
# 
# * Finally, the LASSO regression is the worst of the pack, identifying the tiles but also incorrectly assigning a similar $\theta$ value to many other landscape features.

# #### Shadows

# In[39]:


function_to_plot(6)


# Comparing the ground truth for shadows against the abundance maps of the regressors I notice the following:
# 
# * The **sum-to-one non-negativity** constraint regressors succeeds best in identifying the shadows at the campus. However it assigns a similar $\theta$ value to the torrent as well.
# 
# * The second best performance is that of the **non-negativity** constraint regressor which also identifies the shadows, but incorrectly assigns a similar $\theta$ value to the torrent and the meadows. 
# 
# * The LASSO regression correctly identifies the shadows but incorrectly assigns a similar $\theta$ value to a lot of other landscape features.
# 
# * Finally, the simple ols and the sum-to-one constraint regressors come in last, assigning similar $\theta$ values to most of the campus.

# #### Meadows

# In[40]:


function_to_plot(7)


# Comparing the ground truth for meadows against the abundance maps of the regressors I notice the following:
# 
# * The **non-negativity** constraint regressors succeeds best in identifying the meadows at the campus. However it assigns a similar $\theta$ value to the torrent and some asphalt features as well.
# 
# * The second best performance is that of the **sum-to-one non-negativity** constraint regressor which identifies the meadows, but incorrectly assigns a similar $\theta$ value to the torrent and the meadows. 
# 
# * The LASSO regression correctly identifies the meadows but incorrectly assigns a similar $\theta$ value to a lot of other landscape features.
# 
# * Finally, the simple ols and the sum-to-one constraint regressors incorrectly assign a similar $\theta$ value to most of the campus.

# #### Bare Soil

# In[41]:


function_to_plot(8)


# Comparing the ground truth for bare soil against the abundance maps of the regressors I notice the following:
# 
# * The **non-negativity** constraint regressors succeeds best in representing the bare soil at the campus, while incorrectly assigning a similar $\theta$ value to the torrent and the bitumen. 
# 
# * The second best performance is that of the **sum-to-one non-negativity** constraint regressor which succeeds in representing the bare soil at the campus. However it assigns a similar $\theta$ value to the torrent, the bitumen, some tiles and some other features as well.
# 
# * The simple ols and the sum-to-one constraint regressors correctly identify the bare soil but they incorrectly assign a similar $\theta$ value to a lot of other features.
# 
# * The LASSO regression comes in last, as it doesn't identify the bare soil at all..

# ### Conclusions

# Summing up I get that the following results by comparing the abundance map of each material against the ground truth:
# 
# | Material  | best fit  | second best fit  |
# |:-:|:-:|:-:|
# | Water  |  non-negativity constraint | sum-to-one - non-negativity-constraint  |
# |Trees|  non-negativity constraint  |  sum-to-one - non-negativity-constraint    | 
# |Asphalt   |   non-negativity constraint |  sum-to-one - non-negativity-constraint    |
# | Bricks  |  non-negativity constraint  |  sum-to-one - non-negativity-constraint    |
# |Bitumen   |  non-negativity constraint   | simple ols  | 
# |Tiles   | sum-to-one constraint  |  sum-to-one - non-negativity-constraint    |
# |Shadows   |  non-negativity constraint  | sum-to-one constraint  |
# | Meadows  | non-negativity constraint   |   sum-to-one - non-negativity-constraint   | 
# | Bare Soil  | non-negativity constraint   |  sum-to-one - non-negativity-constraint    |

# Additionally, in the [reconstruction error](#Reconstruction-error) section above, I calculate the following values:
# 
# |Method|Error|
# |---|---|
# |Least squares error|                         0.0015|
# |Least squares sum-to-one error |             0.002|
# |Least squares non-negative error |            0.0042|
# |Least squares sum-to-one non-negative error | 0.0126|
# |Least squares LASSO error |                  0.0101|
# 
# The Least squares non-negativity constraint regression error is in the middle of the variance of the errors. 
# 
# Since it performs the best compared to the other regressors, it comes out on top in this case study.

# # Part 2: Classification

# ## Import data for classification

# In[42]:


# Trainining set for classification 

# import data
Pavia_labels = sio.loadmat('classification_labels_Pavia.mat')

# prepare test set
Test_Set = (np.reshape(Pavia_labels['test_set'],(200,300))).T

# prepare operational set
Operational_Set = (np.reshape(Pavia_labels['operational_set'],(200,300))).T

# prepare training set
Training_Set = (np.reshape(Pavia_labels['training_set'],(200,300))).T


# ## Prepare data for classification

# In[43]:


# compute the indexes in order to exclude zero values

train = np.where(Training_Set != 0)
test = np.where(Test_Set != 0)
operate = np.where(Operational_Set != 0)


# In[44]:


# apply index to exlude zero values
# prepare the endmembers datasets

train_labels = Training_Set[train[0], train[1]]
test_labels = Test_Set[test[0], test[1]]
operate_label = Operational_Set[operate[0], operate[1]]

len(train_labels), len(test_labels), len(operate_label)


# In[45]:


# apply index to exlude zero values
# prepare the pixels datasets

train_hsi = HSI[train[0], train[1],:]
test_hsi = HSI[test[0], test[1],:]
operate_hsi = HSI[operate[0], operate[1]]

len(train_hsi), len(test_hsi), len(operate_hsi)


# In[46]:


# Indices (rows and columns) stored in 2 arrays, corresponding to non-zero labels
# in the train and test sets respectively.
non_zero_train = np.where(Training_Set!=0)
non_zero_test = np.where(Test_Set!=0)

training_labels = Training_Set[non_zero_train[0],non_zero_train[1]]
training_pixels = HSI[non_zero_train[0],non_zero_train[1],:]

test_labels = Test_Set[non_zero_test[0],non_zero_test[1]]
test_pixels = HSI[non_zero_test[0],non_zero_test[1],:]


# ## Plots

# In[47]:


# Plot labels
plt.imshow(labels, cmap = 'CMRmap') # plot the array
plt.colorbar() # plot the color bar


# In[48]:


# Plot test set
plt.imshow(Test_Set, cmap = 'CMRmap') # plot the array
plt.colorbar() # plot the color bar


# In[49]:


# Plot dev set
plt.imshow(Operational_Set, cmap = 'CMRmap') # plot the array
plt.colorbar() # plot the color bar


# In[50]:


# Plot training set
plt.imshow(Training_Set, cmap = 'CMRmap') # plot the array
plt.colorbar() # plot the color bar


# ## Scope

# The task is to assign each one of them to the most appropriate class among the 9 known endmembers (classes). 
# 
# The classification is perfomed with **four** pre-chosen classifiers:
# 
# The [first step](#(A)-Classification-for-each-classifier) is the performance of a 10-fold cross validation for each classifier. For this step the estimated validation error gets reported.
# 
# The second step is the training of each classifier and the evaluation of its performance. This includes the computation of the confusion matrix and the success rate of the classifier.
# 
# [Finally](#(B)-Comparison-of-classifiers) I compare the results of the four classifiers.
# 
# 

# ## (A) Classification for each classifier

# ### Naive Bayes classifier

# #### (i) Cross validation

# In[51]:


cv_naive_bayes = cross_val_score(GaussianNB(), 
                                X = train_hsi, 
                                y = train_labels, 
                                cv = StratifiedKFold(n_splits = 10, 
                                                     shuffle = True))
error_naive_bayes = 1 - cv_naive_bayes

print('The Naive Bayes classifier produces an estimated validation error with mean %.4f and standard deviation %.4f' % 
      (error_naive_bayes.mean(), error_naive_bayes.std()))


# #### (ii) Confusion Matrix & success rate

# In[52]:


# train model, predict for test labels
# calculate the confusion matrix and success rate

model_naive_bayes = GaussianNB().fit(train_hsi, 
                                     train_labels).predict(test_hsi)

naive_confusion_m = confusion_matrix(test_labels,
                                     model_naive_bayes)

success_naive_bayes = naive_confusion_m.trace() / naive_confusion_m.sum()

print('The Naive Bayes classifier has a success rate of %.4f.' % success_naive_bayes)


# In[53]:


# view the confusion matrix

pd.DataFrame(naive_confusion_m, index = [materials_masked[i] for i in range(1, 10)],
             columns = [materials_masked[i] for i in range(1, 10)])


# ### Minimum Euclidean distance classifier

# #### (i) Cross validation

# In[54]:


folds_eucl = StratifiedKFold(n_splits = 10,
                             shuffle = True).split(train_hsi,
                                                   train_labels)

def average_value(endmember, pixel):
    average = np.zeros((9,103))
    for label in range(1,10):
        class_index = np.where(endmember == label)[0]
        class_points  = pixel[class_index]
        class_average = class_points.mean(axis=0)
        average[label - 1,:] = class_average
                               
    return average

errors_euclidean = []

for fold in folds_eucl:
    temp_train_hsi = train_hsi[fold[0]]
    temp_train_labels = train_labels[fold[0]]
    temp_val_pixels = train_hsi[fold[1]]
    temp_val_labels = train_labels[fold[1]]
                                 
    # Average value for each endmember
    # Temporar training set
    temp_average = average_value(temp_train_labels,
                                    temp_train_hsi)
    
    # Distances between temporar training set and validation set.
    temp_distances = distance.cdist(temp_average, 
                                    temp_val_pixels)

    # Predict based on the minimum distance. 

   
    temp_predictions = temp_distances.argmin(axis=0) + 1
    temp_confusion_m = confusion_matrix(temp_val_labels,
                                        temp_predictions)
    
    cv_euclidean = (temp_confusion_m.trace() / temp_confusion_m.sum())
    temp_error = 1 - cv_euclidean
    
    errors_euclidean.append(temp_error)
    
errors_euclidean_ar = np.array(errors_euclidean)
    
print('The minimum Euclidean distance classifer produces an estimated validation error with mean %.4f and standard deviation %.4f' % 
      (errors_euclidean_ar.mean(), errors_euclidean_ar.std()))


# #### (ii) Confusion Matrix & success rate

# In[55]:


# train model, predict for test labels
# calculate the confusion matrix and success rate

average = np.zeros((9,103))
for label in range(1,10):
    class_index = np.where(Training_Set == label)
    class_points  = HSI[class_index[0],class_index[1],:]
    class_average = class_points.mean(axis=0)
    average[label - 1,:] = class_average

distances = distance.cdist(average, test_hsi)
minimum_distances_pred = distances.argmin(axis=0) + 1

euclidean_confusion_m = confusion_matrix(test_labels,
                                         minimum_distances_pred)

success_euclidean = euclidean_confusion_m.trace() / euclidean_confusion_m.sum()

print('The Minimum Euclidean Distance classifier has a success rate of %.4f.' 
      % success_euclidean)


# In[56]:


# view the confusion matrix

pd.DataFrame(euclidean_confusion_m, index = [materials_masked[i] for i in range(1, 10)],
             columns = [materials_masked[i] for i in range(1, 10)])


# ### k-nearest neighbor classifier

# #### (i) Cross validation

# In[57]:


cv_knn = cross_val_score(KNeighborsClassifier(7), 
                                X = train_hsi, 
                                y = train_labels, 
                                cv = StratifiedKFold(n_splits = 10, 
                                                     shuffle = True))
error_kk = 1 - cv_knn

print('The k-nearest neighbor classifier produces an estimated validation error with mean %.4f and standard deviation %.4f' % 
      (error_kk.mean(), error_kk.std()))


# #### (ii) Confusion Matrix & success rate

# In[58]:


# train model, predict for test labels
# calculate the confusion matrix and success rate

neigh_classifier = KNeighborsClassifier(n_neighbors = 7)
model_kneigh = neigh_classifier.fit(train_hsi,
                                    train_labels).predict(test_hsi)

knn_confusion_m = confusion_matrix(test_labels, model_kneigh)

success_knn = knn_confusion_m.trace() / knn_confusion_m.sum()

print('The k-nearest neighbor classifier has a success rate of %.4f.' 
      % success_knn)


# In[59]:


# view the confusion matrix

pd.DataFrame(knn_confusion_m, index = [materials_masked[i] for i in range(1, 10)],
             columns = [materials_masked[i] for i in range(1, 10)])


# ### Bayesian classifier

# #### (i) Cross validation

# In[60]:



cv_bayes = cross_val_score(QuadraticDiscriminantAnalysis(), 
                                X = train_hsi, 
                                y = train_labels, 
                                cv = StratifiedKFold(n_splits = 10, 
                                                     shuffle = True))
error_bayes = 1 - cv_bayes

print('The Bayes classifier produces an estimated validation error with mean %.4f and standard deviation %.4f' % 
      (error_bayes.mean(), error_bayes.std()))


# #### (ii) Confusion Matrix & success rate

# In[61]:


# train model, predict for test labels
# calculate the confusion matrix and success rate

module_bayes = QuadraticDiscriminantAnalysis().fit(train_hsi,
                                                   train_labels)
model_bayes = module_bayes.predict(test_hsi)

bayes_confusion_m = confusion_matrix(test_labels,
                                     model_bayes)

success_bayes = bayes_confusion_m.trace() / bayes_confusion_m.sum()

print('The Bayes classifier has a success rate of %.4f.' % success_bayes)


# In[62]:


# view the confusion matrix

pd.DataFrame(bayes_confusion_m, index = [materials_masked[i] for i in range(1, 10)],
             columns = [materials_masked[i] for i in range(1, 10)])


# ## (B) Comparison of classifiers

# ### Estimated validation error

# In[63]:


print('Naive Bayes: {:>20} = {:>1}, standard deviation = {:>5}' .format('mean',
                                                                        round(error_naive_bayes.mean(),4), 
                                                                        round(error_naive_bayes.std(), 4)))
print('Minimum Euclidean distance : mean = {:>5}, standard deviation = {:>5}' .format(round(errors_euclidean_ar.mean(),4), 
                                                                        round(errors_euclidean_ar.std(), 4)))

print('K-nearest neighbor: {:>13} = {:>1}, standard deviation = {:>5}' .format('mean',
                                                                        round(error_kk.mean(),4), 
                                                                        round(error_kk.std(), 4)))

print('Bayesian: {:>23} = {:>1}, standard deviation = {:>5}' .format('mean',
                                                                     round(error_bayes.mean(),4), 
                                                                     round(error_bayes.std(), 4)))


# The **Bayesian method** has the **smallest** mean value of validation error as well as the smallest deviation.
# 
# After that comes the **K-nearest neighbor** classifier.

# ### Confusion matrices and success rates

# In[64]:


# view the confusion matrix

pd.DataFrame(naive_confusion_m, index = [materials_masked[i] for i in range(1, 10)],
             columns = [materials_masked[i] for i in range(1, 10)])


# In[65]:


print('The Naive Bayes classifier has a success rate of %.4f.' % success_naive_bayes)


# The **Naive Bayes** classifier performs a little more than the average, with a success rate of 66 %.
# I notice that types of land cover such as trees, bricks, bitumen, shadows and bare soil get identified very well, while some other types are misidentified, ie asphalt is very often identified as meadows or tiles are trees.

# In[66]:


# view the confusion matrix

pd.DataFrame(euclidean_confusion_m, index = [materials_masked[i] for i in range(1, 10)],
             columns = [materials_masked[i] for i in range(1, 10)])


# In[67]:


print('The Minimum Euclidean Distance classifier has a success rate of %.4f.' 
      % success_euclidean)


# The **Minimum Euclidean Distance** classifier performs just a bit above 50 %. Most types of land cover are misidentified, with the only exception being the Bare soil.

# In[68]:


# view the confusion matrix

pd.DataFrame(knn_confusion_m, index = [materials_masked[i] for i in range(1, 10)],
             columns = [materials_masked[i] for i in range(1, 10)])


# In[69]:


print('The k-nearest neighbor classifier has a success rate of %.4f.' 
      % success_knn)


# The **k-nearest neighbor** classifier performs very well with a success rate close to 90%. Bare soil is in all case correctly identified, while most of the types of land cover are also correctly attributed to the relevant pixel. The only outliner is the type Meadows that is identified as Asphalt in many cases. 

# In[70]:


# view the confusion matrix

pd.DataFrame(bayes_confusion_m, index = [materials_masked[i] for i in range(1, 10)],
             columns = [materials_masked[i] for i in range(1, 10)])


# In[71]:


print('The Bayes classifier has a success rate of %.4f.' % success_bayes)


# The **Bayes classifier** also performs very well with a success rate close to 90%.  As before, bare soil is in all case correctly identified. All other types of land cover are also correctly attributed to the relevant pixel. The only outliner is again the type Meadows that is identified as Asphalt in many cases.

# The **Bayes** classifier has the most diagonal confusion matrix of all four classifiers and the most zero or very small non-diagonal values.
# 
# Then the **k-nearest neighbor** classifier follows with a bit more non-diagonal non-zero data.

# In[72]:


# report statistical metrics for each classifier

naive_bayes_report = classification_report(test_labels,model_naive_bayes,
                                           output_dict = True)

eucl_report = classification_report(test_labels,minimum_distances_pred,
                                    output_dict = True)

knn_report = classification_report(test_labels,model_kneigh,
                                   output_dict = True)

bayes_report = classification_report(test_labels,model_bayes,
                                     output_dict = True)


# In[73]:


# make dataframes for the reports

df_naive_report = pd.DataFrame(data = naive_bayes_report).transpose()
df_naive_report.columns = pd.MultiIndex.from_product([['Naive Bayes classifier'],df_naive_report.columns])

df_min_euc_report = pd.DataFrame(data = eucl_report).transpose()
df_min_euc_report.columns = pd.MultiIndex.from_product([['Minimum Euclidean distance classifier'],df_min_euc_report.columns])

df_knn_report = pd.DataFrame(data = knn_report).transpose()
df_knn_report.columns = pd.MultiIndex.from_product([['k-nearest neighbor classifier'],df_knn_report.columns])

df_bayes_report = pd.DataFrame(data = bayes_report).transpose()
df_bayes_report.columns = pd.MultiIndex.from_product([['Bayes classifier'],df_bayes_report.columns])


# In[74]:


# combine dataframes

pd.concat([df_naive_report.round(3), df_min_euc_report.round(3), 
           df_knn_report.round(3), df_bayes_report.round(3)], axis = 1)


# In[75]:


report_all = pd.concat([df_naive_report.round(3), df_min_euc_report.round(3), 
           df_knn_report.round(3), df_bayes_report.round(3)], axis = 1)
report_all.iloc[0:9,:].describe()


# Reviewing the report of all for classifiers in regard to all classes I notice that:
# 
# * The **k-nearest neighbor** classifier has the maximum average precision, recall and F-1 score.
# * The **Bayes** classifier has the second maximum average precision, recall and F-1 score.
# * The **Naive Bayes** classifier has the next maximum average precision, recall and F-1 score.
# * The **Minimum Euclidean distance** classifier has the lowest average precision, recall and F-1 score.

# In[76]:


report_all.iloc[9:12,:]


# Reviewing the report of all for classifiers in regard to the overall performance I notice that:
# 
# * The **k-nearest neighbor** classifier has again the maximum average precision, recall and F-1 score for all three categories, ie. accuracy, macro average and weighted average.
# * The **Bayes** classifier has the second maximum average precision, recall and F-1 score in that respect.
# * The **Naive Bayes** classifier has the next maximum average precision, recall and F-1 score.
# * The **Minimum Euclidean distance** classifier has the lowest average precision, recall and F-1 score.

# ### Plots with mixed dataset HSI

# ### Synthesis of plots

# In this section I test trained models eith the mixed signature HSI. For these 109 mixed spectral signatures I predict the labels.
# 
# I plot all abundance maps together and compare the results per material.

# #### Predictions

# **Naive Bayes classifier**

# In[77]:


# predict labels for HSI dataset

HSI_reshp = np.reshape(HSI, (300*200, 103)) # reshape HSI

# predict labels
naive_predict = GaussianNB().fit(train_hsi, 
                                 train_labels).predict(HSI_reshp)

# reshape prediction
naive_predict_reshp = np.reshape(naive_predict, (300,200)) 

# mask zero values
naive_predict_masked = np.ma.masked_equal(naive_predict_reshp, 0) 


# In[78]:


plt.imshow(naive_predict_masked, cmap = cmap_masked)
plt.legend(handles=patches_masked, bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=0. )
plt.colorbar() # plot the color bar


# **Minimum Euclidean distance classifier**

# In[79]:


# predict labels for HSI dataset

HSI_reshp = np.reshape(HSI, (300*200, 103)) # reshape HSI

# predict labels
distances_euclidean = distance.cdist(average, HSI_reshp)
minimum_distances_eucl_pred = distances_euclidean.argmin(axis=0) + 1

# reshape prediction
eucl_predict_reshp = np.reshape(minimum_distances_eucl_pred, (300,200)) 

# mask zero values
eucl_predict_reshp_masked = np.ma.masked_equal(eucl_predict_reshp, 0) 


# In[80]:


plt.imshow(eucl_predict_reshp_masked, cmap = cmap_masked)
plt.legend(handles=patches_masked, bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=0. )
plt.colorbar() # plot the color bar


# **k-nearest neighbor classifier**

# In[81]:


# predict labels for HSI dataset

HSI_reshp = np.reshape(HSI, (300*200, 103)) # reshape HSI

# predict labels
neigh_predict = neigh_classifier.fit(train_hsi,train_labels).predict(HSI_reshp)

# reshape prediction
neigh_predict_reshp = np.reshape(neigh_predict, (300,200)) 

# mask zero values
neigh_predict_masked = np.ma.masked_equal(neigh_predict_reshp, 0) 


# In[82]:


plt.imshow(neigh_predict_masked, cmap = cmap_masked)
plt.legend(handles=patches_masked, bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=0. )
plt.colorbar() # plot the color bar


# **Bayes classifier**

# In[83]:


# predict labels for HSI dataset

HSI_reshp = np.reshape(HSI, (300*200, 103)) # reshape HSI

# predict labels
bayes_predict = module_bayes.predict(HSI_reshp)

# reshape prediction
bayes_predict_reshp = np.reshape(bayes_predict, (300,200)) 

# mask zero values
bayes_predict_reshp_masked = np.ma.masked_equal(bayes_predict_reshp, 0) 


# In[84]:


plt.imshow(bayes_predict_reshp_masked, cmap = cmap_masked)
plt.legend(handles=patches_masked, bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=0. )
plt.colorbar() # plot the color bar


# In[85]:


# set a function to plot each label in order to compare on a
# one-by-one basis

def function_to_plot_classifiers(endmember):

    fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(figsize = (30,20), nrows = 1, ncols = 5)

    divider5 = make_axes_locatable(ax5)
    
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    
    im1 = ax1.imshow(np.ma.masked_where(naive_predict_masked != (endmember + 1), 
                                        naive_predict_masked), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    im2 = ax2.imshow(np.ma.masked_where(eucl_predict_reshp_masked != (endmember + 1), 
                                        eucl_predict_reshp_masked), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    im3 = ax3.imshow(np.ma.masked_where(neigh_predict_masked != (endmember + 1), 
                                        neigh_predict_masked), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    im4 = ax4.imshow(np.ma.masked_where(bayes_predict_reshp_masked != (endmember + 1), 
                                        bayes_predict_reshp_masked), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    
    im5 = ax5.imshow(np.ma.masked_where(labels != (endmember + 1), 
                                        labels), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
 
    fig.colorbar(im5, cax=cax5, orientation='vertical')    
        
    
    #plt.subplots_adjust(right = 1)
    
    ax1.set_title('Naive Bayes\nclassifier', 
                  fontsize = 25)
    ax2.set_title('Minimum Euclidean\ndistance classifier', fontsize = 25)
    ax3.set_title('k-nearest neighbor\nclassifier', fontsize = 25)
    ax4.set_title('Bayes classifier', fontsize = 25)
    ax5.set_title('Ground truth for\nendmember %d' % 
                  ((endmember+1)), fontsize = 25)
        # ax6.legend(handles=patches_masked, 
        #           bbox_to_anchor=(1.3, 1), 
        #           loc=2, borderaxespad=0.)
    fig.suptitle('Reconstructed abundance maps vs ground truth (Label %s) for %s' % 
                 ((endmember+1), materials_masked[endmember+1]), fontsize = 30, y= 0.75)
    
    


# #### Water

# In[86]:


function_to_plot_classifiers(0)


# Comparing the ground truth for water against the abundance maps of the classifiers I notice the following:
# 
# * The **Bayes** classifier succeeds in representing the torrent diagonally crossing the campus, while assigning label 1 to very few other landscape features. 
# * The second best performance is of the **k-nearest neighbor** classifier, which also identifies the torrent, while incorrectly assigning label 1 to a few other landscape features. 
# * The **Naive Bayes** classifier identifies the torrent but incorrectly assigns label 1 to a lot of other landscape features.
# * Finally, the **Minimum Euclidean distance** classifier performs the worst; it identifies the torrent, but incorrectly assigns label 1 to many other landscape features.

# #### Trees

# In[87]:


function_to_plot_classifiers(1)


# Comparing the ground truth for trees against the abundance maps of the classifiers I notice the following:
# 
# * The **k-nearest neighbor** classifier succeeds in representing the trees at the north of the campus, while assigning label 2 to very few other landscape features. 
# * The second best performance is of the **Bayes** classifier, which also identifies the trees, while incorrectly assigning label 2 to a few other landscape features. 
# * The **Naive Bayes** classifier identifies the trees but incorrectly assigns label 2 to a lot of other landscape features.
# * Finally, the **Minimum Euclidean distance** classifier performs the worst. It identifies the trees, but incorrectly assigns label 2 to many other features.

# #### Asphalt

# In[88]:


function_to_plot_classifiers(2)


# Comparing the ground truth for asphalt against the abundance maps of the classifiers I notice the following:
# 
# * The **k-nearest neighbor** classifier succeeds in representing the network of roads (asphalt) at the west of the campus and only assigns label 3 to very few other landscape features. 
# * The second best performance is of the **Bayes** classifier, which also identifies the roads (asphalt), while incorrectly assigning label 3 to few other landscape features. 
# * The **Minimum Euclidean distance** classifier identifies very few roads (asphalt) and it incorrectly assigns label 3 to a lot of other landscape features.
# * Finally, the **Naive Bayes** classifier performs the worst in identifying the roads (asphalt) and also incorrectly assigns label 3 to many other landscape features.

# #### Bricks

# In[89]:


function_to_plot_classifiers(3)


# Comparing the ground truth for bricks against the abundance maps of the classifiers I notice the following:
# 
# * All four classifiers perform poorly on the bricks label. While they identify the bricks, they all misidentify parts of the roads, meadows and torrent as bricks.

# #### Bitumen

# In[90]:


function_to_plot_classifiers(4)


# Comparing the ground truth for bitumen against the abundance maps of the classifiers I notice the following:
# 
# * The **Bayes** and **k-nearest neighbor** classifiers perform the best; they only assign label 5 to the elements that the ground truth dictates as correct. 
# * The **Minimum Euclidean distance** classifier also identifies almost all pixels with label 5 but it also incorrectly lables other landscape features as bitumen.
# * Finally, the **Naive Bayes** classifier performs the worst. It identifies almost all pixels with label 5 but incorrectly assigns label 5 to many other features.

# #### Tiles

# In[91]:


function_to_plot_classifiers(5)


# Comparing the ground truth for tiles against the abundance maps of the classifiers I notice the following:
# 
# * The **Bayes** classifier succeeds in identifying the tiles but it assigns label 6 to other landscape features too, i.e. it misinterprets the signal.
# * The second best performance is of the **k-nearest neighbor classifier** classifier, which also identifies the tiles, while incorrectly assigning label 6 to a few other landscape features. 
# * The **Naive Bayes** classifier identifies very few tiles and it incorrectly assigns label 6 to a lot of other landscape features.
# * Finally, the **Minimum Euclidean distance** classifier performs the worst by identifying the least of the tiles and incorrectly assigning label 6 to many other landscape features.
# 
# **<span style="font-size:1.2em; color: black">In all four cases, the result in identifying the tiles is not satisfactory.</span>**

# #### Shadows

# In[92]:


function_to_plot_classifiers(6)


# Comparing the ground truth for shadows against the abundance maps of the classifiers I notice the following:
# 
# * THe best performance is that of the **Bayes** classifier, which identifies the shadows with less accuracy than that of the **k-nearest neighbor**, but it doesn't misidentify any other landscape feature.
# * The **k-nearest neighbor** classifier succeeds in representing the shadow almost to its full extent, while assigning label 7 to very few other lanscape features. 
# * The third best performance is of the **Minimum Euclidean distance** classifier, which also identifies the shadows with less accuracy than that of the  **Bayes** and **k-nearest neighbor** classifierts, but incorrectly asigns label 7 to many other lanscape features.
# * Finally, the **Naive Bayes** classifier performs the worst. It identifies the shadows but incorrectly assigns label 7 to many other lanscape features.

# #### Meadows

# In[93]:


function_to_plot_classifiers(7)


# Comparing the ground truth for meadows against the abundance maps of the classifiers I notice the following:
# 
# * The **Bayes** classifier succeeds in representing the meadows of the campus to their full extent and only assigns label 8 to very few other landscape features. 
# * The second best performance is of the **k-nearest neighbor** classifier, which also fully identifies the meadows, while incorrectly assigning label 8 to a few other landscape features. 
# * The **Naive Bayes** classifier fully identifies the meadows, but it incorrectly assigns label 8 to a lot of other landscape features.
# * Finally, the **Minimum Euclidean distance** classifier performs the worst. It fully identifies the meadowsbut incorrectly assigns label 8 to many other landscape features.

# #### Bare Soil

# In[94]:


function_to_plot_classifiers(8)


# Comparing the ground truth for bare soil against the abundance maps of the classifiers I notice the following:
# 
# * The **Naive Bayes** and **Bayes** classifiers perform equally well. They identify the bare soil to its full extent and only assign label 9 to a few other landscape features. 
# * The second best performance is of the **Minimum Euclidean distance** and **k-nearest neighbor classifier** classifiers, which also fully identify the bare soil, while incorrectly assigning the label 9 to many other landscape features. 
# 
# **<span style="font-size:1.2em; color: black">None of the four classifiers are satisfactory in identifying the tiles.</span>**

# **<span style="font-size:1.3em; color: black">Summing up, taking into account the estimated validation error, the correlation matrices and the abundance maps of the four classifier I conclude that the "winner" is the Bayes classifier, followed by the k-nearest neighbor classifier.</span>**

# # Part 3: Combination - Correlation

# <span style="font-size:1.2em; color: black">I plot the abundance maps of the regressors and classifiers, against the ground truth and discuss the results.</span>

# ## Synthesis of plots

# In[95]:


# set a function to plot each label in order to compare on a
# one-by-one basis

def plot_reg_classifiers(endmember):
    
    fig, [[ax1, ax2, ax3, ax4, ax5], [ax6, ax7, ax8, ax9, ax10]] = plt.subplots(figsize = (30,20), 
                                                                             nrows = 2, 
                                                                             ncols = 5)
    
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    divider3 = make_axes_locatable(ax3)
    divider4 = make_axes_locatable(ax4)    
    divider5 = make_axes_locatable(ax5)    
    divider6 = make_axes_locatable(ax6)
    divider7 = make_axes_locatable(ax7)
    divider8 = make_axes_locatable(ax8)
    divider9 = make_axes_locatable(ax9)    
    divider10= make_axes_locatable(ax10)
    
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    cax6 = divider6.append_axes('right', size='5%', pad=0.05)
    cax7 = divider7.append_axes('right', size='5%', pad=0.05)
    cax8 = divider8.append_axes('right', size='5%', pad=0.05)
    cax9 = divider9.append_axes('right', size='5%', pad=0.05)
    cax10 = divider10.append_axes('right', size='5%', pad=0.05)
    
    im1 = ax1.imshow(thetas_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember+1])
    im2 = ax2.imshow(thetas_sum_one_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember+1]) 
    im3 = ax3.imshow(thetas_non_neg_ar_resh[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember+1])    
    im4 = ax4.imshow(thetas_sum_one_non_neg_ar_resh[:,:,endmember], 
                  cmap = cmaps_shuffle[endmember+1]) 
    
    im5 = ax5.imshow(lasso_abund_map[:,:,endmember], 
                     cmap = cmaps_shuffle[endmember+1])  

    im6 = ax6.imshow(np.ma.masked_where(naive_predict_masked != (endmember + 1), 
                                        naive_predict_masked), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    im7 = ax7.imshow(np.ma.masked_where(eucl_predict_reshp_masked != (endmember + 1), 
                                        eucl_predict_reshp_masked), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    im8 = ax8.imshow(np.ma.masked_where(neigh_predict_masked != (endmember + 1), 
                                        neigh_predict_masked), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    im9 = ax9.imshow(np.ma.masked_where(bayes_predict_reshp_masked != (endmember + 1), 
                                        bayes_predict_reshp_masked), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    

    
    im10 = ax10.imshow(np.ma.masked_where(labels != (endmember + 1), 
                                        labels), 
                     cmap = ListedColormap([color_dict[endmember + 1]]))
    
    fig.colorbar(im10, cax=cax10, orientation='vertical')   
        
    ax1.set_title(r'Linear Reg.: No constraint', 
                  fontsize = 25)
    ax2.set_title('Sum-to-one', fontsize = 25)
    ax3.set_title('Non-negativity', fontsize = 25)
    ax4.set_title('Sum-to-one\n non-negativity', fontsize = 25)
    ax5.set_title('LASSO', fontsize = 25)   
    ax6.set_title('Naive Bayes\nclassifier', 
                  fontsize = 25)
    ax7.set_title('Minimum Euclidean\ndistance classifier', fontsize = 25)
    ax8.set_title('k-nearest neighbor\nclassifier', fontsize = 25)
    ax9.set_title('Bayes classifier', fontsize = 25)    
    ax10.set_title('Ground truth \nfor endmember %d' % 
                  ((endmember+1)), fontsize = 25)
    
    
    fig.suptitle('Reconstructed abundance maps for regressors and classifiers\nvs ground truth (Label %s) for %s' % 
                 ((endmember+1), materials_masked[endmember+1]), fontsize = 30, y= 0.95)
    
    


# ### Water

# In[96]:


plot_reg_classifiers(0)


# Comparing the ground truth for water against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **linear regression** with the **non-negativity** constraint for the $\theta$ values delivers the best result in identifying the torrent in the campus.
# * The **Bayes classifier** delivers a good result in identifying the torrent in the campus by assigning the label 1 to more pixels along the torrent, but suffers from additional noise.
# 
# A combination of these results could lead to an optimal identification of the course and width of the torrent. In this case I could mask the abundance map of the Bayes classifier with the zero values of the linear regression's abundance map, and obtain the optimum result.

# ### Trees

# In[97]:


plot_reg_classifiers(1)


# Comparing the ground truth for trees against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **linear regression** with the **non-negativity** constraint for the $\theta$ values delivers the best result in identifying the torrent in the campus. It misidentifies part of the tiles in the east and some landscape features in the south-west of the campus,
# * The **k-nearest neighbor** classifier achieves a good result in identifying the trees in the campus. However it misidentifies some landscape features in the north-west and south-west as trees. 
# * The south-west corner is misidentified by both models.
# 
# A combination of these results could lead to the best result with regard to getting read of the misfits at the north-west of the campus. In this case I could mask the abundance map of the linear regression with the **non-negativity** constraint with the zero values of the **k-nearest neighbor** classifier's abundance map, and take the optimum result.

# ### Asphalt

# In[98]:


plot_reg_classifiers(2)


# Comparing the ground truth for roads (asphalt) against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **linear regression** with the **non-negativity** constraint for the $\theta$ values delivers the best result in identifying the network of roads (asphalt). However, it misidentifies part of the tiles at the east and the meadows at the north and south of the campus,
# * The **k-nearest neighbor** classifier delivers a good result in identifying the asphalt in the campus but it misidentifies some feature in the west. 
# 
# A combination of these results could lead to the best result with regard to getting read of the misfits at the north-west of the campus. In this case I could mask the abundance map of the linear regression with the **non-negativity** constraint with the zero values of the **k-nearest neighbor** classifier's abundance map, and take the optimum result.

# ### Bricks

# In[99]:


plot_reg_classifiers(3)


# Comparing the ground truth for bricks against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **linear regression** with the **non-negativity** constraint for the $\theta$ values delivers the best result in identifying the bricks. 
# 
# All other results have major errors.

# ### Bitumen

# In[100]:


plot_reg_classifiers(4)


# Comparing the ground truth for bitumen against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **linear regression** with the **non-negativity** constraint for the $\theta$ values delivers the best result in identifying the bitumen.
# * The **k-nearest neighbor** classifier delivers also a very good fit when identifying the bitumen in the campus.

# ### Tiles

# In[101]:


plot_reg_classifiers(5)


# Comparing the ground truth for tiles against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **linear regression** with the **non-negativity** constraint for the $\theta$ values delivers the best result in identifying the tiles. 
# 
# All other results have major errors.

# ### Shadows

# In[102]:


plot_reg_classifiers(6)


# Comparing the ground truth for shadows against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **Bayes** classifier delivers the best and optimum result in identifying the shadows. 

# ### Meadows

# In[103]:


plot_reg_classifiers(7)


# Comparing the ground truth for meadows against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **Bayes** classifier delivers the best and optimum result in identifying the meadows. 

# ### Bare Soil

# In[106]:


plot_reg_classifiers(8)


# Comparing the ground truth for the bare soil against the abundance maps of the regressors and classifiers I notice the following:
# 
# * The **linear regression** with the **non-negativity** constraint for the $\theta$ values delivers the best result in identifying the bare soil. 
# * The **Bayes** classifier is also close to ground truth but misidentifies the bitumen in the north-west of the campus. An advantage of the Bayes classifier is that it provides better information with regard to the geometry of the bare soil stripes.
