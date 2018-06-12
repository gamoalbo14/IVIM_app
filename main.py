#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:06:49 2018

@author: GamoHuertaUrrea
"""


from tkinter.filedialog import askopenfilenames
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

import cv2

from scipy.optimize import least_squares
from math import log




def nifty2vol(fileNames):

    lst=list(fileNames)
    im_tmp=nib.load(lst[0])
    Array=im_tmp.get_data()
    sz=im_tmp.shape
    Array_DWI=np.zeros((sz[0], sz[1], sz[2]*2))
    
    for i in range(len(ivim_fileNames)-1):
        j=i+1
        im_file=nib.load(lst[j])
        im=im_file.get_data()            
        Array=np.concatenate((Array,im),axis=2)
    
    Array_DWI=np.concatenate((Array[:,:,0:sz[2]], Array[:,:,-sz[2]:]), axis=2)
            
    return(Array, Array_DWI)
    
def mask_vol(fileNames,full_vol):
    
    tmp_vol=nib.load(fileNames[0]).get_data()
   
    vol_mask=tmp_vol
    sz=tmp_vol.shape

    T=int(vol_mask.max()*0.1)
    for i in range(sz[0]):
       for j in range (sz[1]):
 
           for k in range(sz[2]):
                vol_mask[i,j,k]=1 if tmp_vol[i,j,k] >= T else 0

    kernel=np.ones((3,3),np.uint8)
    vol_mask_clo=cv2.morphologyEx(vol_mask,cv2.MORPH_CLOSE, kernel, iterations=8)
    #plt.imshow(vol_mask_clo[:,:,12],cmap='gray')
    
    
    sz_full=full_vol.shape
    sz_vol=tmp_vol.shape
    ivim_bvol=full_vol
    dwi_bvol=np.zeros((sz[0], sz[1], sz[2]*2))
    
    
    for i in range(0,sz_vol[2],sz_full[2]):
        ivim_bvol[:,:,i:i+sz_vol[2]:]=vol_mask_clo*full_vol[:,:,i:i+sz_vol[2]]
        
    dwi_bvol=np.concatenate((ivim_bvol[:,:,0:sz[2]], ivim_bvol[:,:,-sz[2]:]), axis=2)    
    
    return (vol_mask_clo, ivim_bvol, dwi_bvol)


def difffitmonoexp(dataADC,coeff):
    
    #x(1) = constante inicial
    #x(2) = ADC
    
    SIfit1 =np.dot(coeff[0],(np.exp(np.dot(-dataADC[1,:],coeff[1]))))
    
    return SIfit1


def difffitbiexp(dataADC,coeff,d):
    #datadiff(1,:)= SI
    #datadiff(2,:)=b
    #x(1) = constante inicial
    #x(2) = f
    #x(3) = D*
    #x(4) = D
    
    SIfit2 =  (np.dot(coeff[1], np.exp(np.dot(-dataADC[0,:], coeff[2]))))+ np.dot((1-coeff[1]), np.exp(np.dot(-dataADC[1,:],d[1])))
    
    return SIfit2


def residuals(coeffs, x, y):
    return np.power((difffitmonoexp(x, coeffs)-x[0,:]),2)


def residuals2(coeffs, x, y, d):
    return np.power((difffitbiexp(x, coeffs,d)-x[0,:]),2)


def plot_im(ADC_im, D_im, f_im, Ds_im):
    
    fig = plt.figure()

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(ADC_im[:,:,12], cmap='gray')
    ax1.set_title('ADC map')
    ax1.axis('off')

    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(D_im[:,:,12], cmap='gray')
    ax2.set_title('D map')
    ax2.axis('off')

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(f_im[:,:,12], cmap='gray')
    ax3.set_title('f map')
    ax3.axis('off')

    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(Ds_im[:,:,12], cmap='gray')
    ax4.set_title('D* map')
    ax4.axis('off')
    
    return fig

def select_ROI(ADC_im, D_im, f_im, Ds_im):
    
    im_ADC=ADC_im[:,:,12]
    cv2.namedWindow('real_image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('real_image', (1500,1500))
    r =cv2.selectROI('real_image', im, False, True)
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow('image', imCrop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    


# IVIM study selection - input
ftypes2 = [('NIfTY file',"*.nii")]
ttl2 = "Select All IVIM files"
dir2 = '/Users/GamoHuertaUrrea/'

ivim_fileNames = askopenfilenames(filetypes = ftypes2, initialdir = dir2, title = ttl2)

# Reading and making ivim volume

[ivim_vol, dwi_vol]=nifty2vol(ivim_fileNames)

# bin mask generation and difussion mask
[ivim_binmask, ivim_bvalmask, dwi_bvalmask]= mask_vol(ivim_fileNames, ivim_vol)


#Get ADC & IVIM
bval=[0, 10, 40, 50, 60, 150, 160, 170, 190, 200, 260, 440, 560, 600, 700, 980, 1000]
sz=ivim_binmask.shape
szIV=ivim_vol.shape

ADC_im=np.zeros((sz[0],sz[1],sz[2]), dtype='float64')
D_im=np.zeros((sz[0],sz[1],sz[2]), dtype='float64')
f_im=np.zeros((sz[0],sz[1],sz[2]), dtype='float64')
Ds_im=np.zeros((sz[0],sz[1],sz[2]), dtype='float64')

dataADC=np.zeros((2,len(bval)), dtype='float64')

# Adjust two-steps
for i in range(sz[0]):
    for j in range(sz[1]):
        for k in range(sz[2]):
            
            if dwi_bvalmask[i,j,k] > 0:
                #print('entro')
                #print('valor:',  dwi_bvalmask[i,j,k+20])
                ADC_im[i,j,k] = (log(dwi_bvalmask[i,j,k]/dwi_bvalmask[i,j,k+20])/(bval[-1] -bval[0]))*1000
                l=0
                ind= np.zeros((len(bval)))
                
                
                for a in range(k,szIV[2],sz[2]):
                    ind[l]=ivim_bvalmask[i,j,a]
                    l=l+1
                    
                dataADC[0,:]=ind
                dataADC[1,:]=bval
                dataADC1=dataADC[:,9:17]
                dataADC2=dataADC[:,0:9]
                y=0
                X0= np.array([0,0.8e-3])
                
                
                sol1 = least_squares(residuals, X0, method='lm', args=(dataADC1,y))
                
                X01=np.zeros((3))                
                X01[0]=sol1.x[0]
                X01[1]=0.1
                X01[2]=8e-3
                
                sol2=least_squares(residuals2, X01, method= 'lm', args=(dataADC2,y,sol1.x))
                
                res=np.zeros((4))
                res[0]=sol2.x[0]
                res[3]=sol1.x[1]
                res[1]=sol2.x[1]*100
                res[2]=sol2.x[2]*1000
                res[3]=res[3]*1000
                
                D_im[i,j,k]= res[3]
                f_im[i,j,k]= res[1]
                Ds_im[i,j,k]= res[2]
                #print('procesando')
                
plot_im(ADC_im, D_im, f_im, Ds_im)

#select_ROI()


im_ADC = ADC_im[:,:,12]
im_D = D_im[:,:,12]
im_f = f_im[:,:,12]
im_Ds = Ds_im[:,:,12]

cv2.namedWindow('real_image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('real_image', (1500,1500))
r = cv2.selectROI('real_image', im_ADC, False, True)

imCrop_ADC = im_ADC[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
imCrop_D = im_D[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
imCrop_f = im_f[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
imCrop_Ds = im_Ds[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

#cv2.imshow('image', imCrop_ADC)
cv2.waitKey(0)
cv2.destroyAllWindows()

mn_ADC = round(imCrop_ADC.mean(),2)
mn_D = round(imCrop_D.mean(),2)
mn_f = round(imCrop_f.mean(),2)
mn_Ds = round(imCrop_Ds.mean(),2)

std_ADC = round(imCrop_ADC.std(),2)
std_D = round(imCrop_D.std(),2)
std_f = round(imCrop_f.std(),2)
std_Ds = round(imCrop_Ds.std(),2)

c = 'mean ADC:' + ' ' + str(mn_ADC) + '+-' + str(std_ADC) + '\n' + 'mean D:' + ' ' + str(mn_D) + '+-' + str(std_D) + '\n' + 'mean f:' + ' ' + str(mn_f) + '+-' + str(std_f) + '\n' + 'mean Ds:' + ' ' + str(mn_Ds) + '+-' + str(std_Ds)
      
tk.messagebox.showinfo("info name", c)




        
                
                
                
            
    
            
            









       



            
            


