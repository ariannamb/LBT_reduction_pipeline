##### LBT data reduction:

##Import some basic functions:
import sys
import os
import numpy as np
from astropy.io import fits
import progressbar
from progressbar import ProgressBar
from datetime import datetime
import datetime
from numba import prange
from find_stars import *
from classicalADI import *
import resource
import multiprocessing
from functools import partial
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans
import gc
import pickle

import time
start = time.time()

##Set the path to the master flat:
master_flat=fits.open('master_flat_2.fits')[0].data
##Set the bad pixel map to use:
Bad_Pixel_map=fits.open('bad_pixels_mask_0.25.fits')[0].data
##set the paths where to save the final DX and SX stamps:
savepath_DX='/reduced/SCIENCE/DX/'
savepath_SX='/reduced/SCIENCE/SX/'
savepath_DX_centre='/reduced/SCIENCE/DX_fcentre/'
savepath_SX_centre='/SCIENCE/SX_fcentre/'
##Set the Verbose option
target_name='target'
verbose=True
quick_reduction=True #set this to 'true' to get a quick ADI reduction at the end (i.e: classical ADI)
###set the path to the science frames and the threshold
path_to_frames='/raw/SCIENCE/'
threshold=3700
##Set the expected number of stars per frame (i.e: 1 dish or 2 dish observation), and the expected nod:
n_stars=2
up_down=False #False = 'left'-'right' nodding and True = 'up'-'down' nodding, for the rest of the script, nodA referes to either 'up' or 'left' and nodB to 'down' and ''right
sep_ud= 500 ##separation, in pixel value, between up-down
sep_lf= 1060 ##separation, in pixel value, between left-right
##Set the maximum time distance between two consecutive skies (i.e: all frames that differ less than this amount in time will be median combined together to create a single sky)
diff=1 ##considering that the typical time difference between two similar nods is ~3 minutes, I set a lower threshold for sky variations, 100 seconds sounds reasonable
###Set the desired size (in pixels) of the final reduced frames
size_final=500
frame_index='DX_2' ##this tells if the frame_DX uses the cut_indexes with suffix 2 or 1 (i.e: a1,b1,c1,d1,a2,b2,c2,d2 used for padding the images before dewarping them)
shape_data=[1024,2048] ##size of the window for these observations

##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
### define the find_stars function here, so that it can be parallelized:
def find_stars_parell(path,threshold,type,Bad_Pixel_map,unsat=False):
    '''
    This function finds the position of the star(s) in a given frame as the maxima above a given set threshold and return them as an x,y position.
    A bad pixel map is passed as a parameter to ignore bad pixels.
    '''
    #######
    #import the image as an array
    if type=='fits':
        image=fits.open(path,memmap=False)
        array=image[0].data
        image.close()
    if type=='array':
        array=np.array(path)
    ###
    if unsat==True:
        ###Now, let's ignore the bad pixel while looking for the star(s), equating the bad pixels to zero:
        mask=np.isnan(Bad_Pixel_map) ##get the mask
        array=np.array(array,dtype=float)
        array[mask]=0 #set the bad pixels to zero
    ##and now smooth the image and find the local maxima (i.e.: the star(s)):
    image_smooth=scipy.ndimage.gaussian_filter(array,sigma=2) #smooth the image with a gaussian of sigma =2
    maxima=(image_smooth>threshold) #find all the maxima (i.e: the star) of the smoothed image_path
    x=np.where(maxima==True)[0] #evaluate the mean position of all the maxima
    y=np.where(maxima==True)[1]  #evaluate the mean position of all the maxima
    ####Now let's find how many maxima I have in this image:
    if len(x) > 0:
        x_all=[]
        y_all=[]
        for i in range(len(x)-1):
            if x[i] == x[i+1] or x[i] == x[i+1] -1:
                pass
            else:
                x_all.append(x[0:i])
                x_all.append(x[i+1:])
                y_all.append(y[0:i])
                y_all.append(y[i+1:])
        ##In the case in which there is only one star found
        if len(x_all)==0:
            x_all.append(x[:])
            y_all.append(y[:])
        ###Let's return the mean x and y position for every maximum (i.e: every star) in the image:
        stars=[]
        for l in range(len(x_all)):
            stars.append([np.mean(x_all[l]), np.mean(y_all[l])])
        return np.array(stars)
        #print 'The image has ',len(x_all),' stars, with positions: ',star
    if len(x)==0:
        return np.array([])
##########################################################################################################################################################
##########################################################################################################################################################

##this performs a principal component approach based- sky subtraction:
def pca_subtraction(master_flat,skynod,datanod,up_down,savepath_DX,savepath_SX,n_stars,Bad_Pixel_map,std=2):
    '''
    Given a datacube of sky frames, this function performs a principal components based approach-based sky subtraction, 
    and saves the sky-subtracted frames in the specified location.
    '''
    skycube=[]
    for i in range(len(skynod.keys())):
        skycube.append(skynod[str(sorted(skynod.keys())[i])][2])
    datacube=[]
    for i in range(len(datanod.keys())):
        datacube.append(datanod[str(sorted(datanod.keys())[i])][2])

    truncate_pca=len(skycube)
    #obj = skycube
    nobj=len(skycube)
    skycube_mean=np.mean(skycube,axis=(1,2))
    skycube=skycube-skycube_mean[:,None,None]
    data=np.reshape(skycube,(nobj,np.shape(skycube)[1]*np.shape(skycube)[2]))
    del skycube
    covMatrix=np.dot(data,data.T)
    eigenval,eigenvect=np.linalg.eigh(covMatrix)
    del covMatrix
    pc_orig=np.dot(eigenvect.T,data)
    del data
    pc= pc_orig/(eigenval[:,None])
    datacube_mean=np.mean(datacube,axis=(1,2))
    datacube=datacube-datacube_mean[:,None,None]
    data2=np.reshape(datacube,(len(datacube),np.shape(datacube)[1]*np.shape(datacube)[2]))
    s1=np.dot(pc_orig,data2.T)
    del data2
    sk=np.dot(s1.T,pc)
    del s1
    gc.collect()
    sk_cube=np.reshape(sk,(len(sk),np.shape(datacube)[1],np.shape(datacube)[2]))
    sky_subtracted=datacube-sk_cube
    del sk_cube

    ##get the star(s) position(s):
    stars_pos=[datanod[f][0] for f in sorted(datanod.keys())]
    ##get the corresponding names:
    names=[f for f in sorted(datanod.keys())]

    ##get the mask:
    mask=np.isnan(Bad_Pixel_map)

    ##get the rough sizes:
    if n_stars>1:
        if up_down==True:
            rough_sizes=[np.int((np.abs(stars_pos[f][0][1] - stars_pos[f][1][1])) - 20) for f in range(len(stars_pos))]
        if up_down==False:
            rough_sizes=[np.int((np.abs(stars_pos[f][0][0] - stars_pos[f][1][0])) - 20) for f in range(len(stars_pos))]
    if n_stars==1:
        rough_sizes=[np.int((np.min([len(datacube[f][2])-stars_pos[f][0][1],len(datacube[f][1])-stars_pos[f][0][0]])) -22) for f in range(len(stars_pos))]
    ##now, let's find the limits for the stamps:
    if n_stars>1:
        a1=np.array([np.int(stars_pos[f][0][0]) - np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        b1=np.array([np.int(stars_pos[f][0][0]) + np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        c1=np.array([np.int(stars_pos[f][0][1]) - np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        d1=np.array([np.int(stars_pos[f][0][1]) + np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        a2=np.array([np.int(stars_pos[f][1][0]) - np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        b2=np.array([np.int(stars_pos[f][1][0]) + np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        c2=np.array([np.int(stars_pos[f][1][1]) - np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        d2=np.array([np.int(stars_pos[f][1][1]) + np.int(rough_sizes[f]) for f in range(len(stars_pos))])

        a1[a1 < 0] = 0
        b1[b1 < 0] = 0
        c1[c1 < 0] = 0
        d1[d1 < 0] = 0
        a2[a2 < 0] = 0
        b2[b2 < 0] = 0
        c2[c2 < 0] = 0
        d2[d2 < 0] = 0
        b1[b1> np.shape(datacube[0])[0]]=np.shape(datacube[0])[0]
        b2[b2> np.shape(datacube[0])[0]]=np.shape(datacube[0])[0]
        d1[d1> np.shape(datacube[0])[1]]=np.shape(datacube[0])[1]
        d2[d2> np.shape(datacube[0])[1]]=np.shape(datacube[0])[1]
        ###create a dictionary that links names and cut indexes:
        dict_cut=dict(zip(names, zip(a1,b1,c1,d1,a2,b2,c2,d2)))

    if n_stars==1:
        a=np.array([np.int(stars_pos[f][0][0]) - np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        b=np.array([np.int(stars_pos[f][0][0]) + np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        c=np.array([np.int(stars_pos[f][0][1]) - np.int(rough_sizes[f]) for f in range(len(stars_pos))])
        d=np.array([np.int(stars_pos[f][0][1]) + np.int(rough_sizes[f]) for f in range(len(stars_pos))])

        a[a < 0] = 0
        c[c < 0] = 0
        b[b> np.shape(datacube[0])[0]]=np.shape(datacube[0])[0]
        d[d> np.shape(datacube[0])[1]]=np.shape(datacube[0])[1]
        ###create a dictionary that links names and cut indexes:
        dict_cut=dict(zip(names, zip(a,b,c,d)))

    ##delete stuff to make space:
    del datacube, sk
    gc.collect()
    #import IPython
    #IPython.embed()
    ##now let's go frame by frame, divide by the master flat, cut into DX and SX and save in the corresponding folders:
    pbar=ProgressBar()
    for i in pbar(range(len(sky_subtracted))):
        frame_i=np.array(sky_subtracted[i]/master_flat)
        ##mask the bad pixels with Nans (they will be corrected in later stage):
        frame_i[mask]=np.nan
        if n_stars>1:
            frame_1=frame_i[a1[i]:b1[i],c1[i]:d1[i]]
            frame_2=frame_i[a2[i]:b2[i],c2[i]:d2[i]]
            if up_down==True:
                if stars_pos[10][0][1] < stars_pos[10][1][1]:
                    frame_DX=frame_1
                    frame_SX=frame_2
                if stars_pos[10][0][1] > stars_pos[10][1][1]:
                    frame_SX=frame_1
                    frame_DX=frame_2
            if up_down==False:
                if stars_pos[10][0][0] > stars_pos[10][1][0]:
                    frame_DX=frame_1
                    frame_SX=frame_2
                if stars_pos[10][0][0] < stars_pos[10][1][0]:
                    frame_SX=frame_1
                    frame_DX=frame_2
            ##save the frames in the correct folder with the correct name:
            frame_DX_save=fits.PrimaryHDU(frame_DX)
            frame_DX_save.writeto(savepath_DX+str(names[i]))
            frame_SX_save=fits.PrimaryHDU(frame_SX)
            frame_SX_save.writeto(savepath_SX+str(names[i]))
        if n_stars==1:
            frame=frame_i[a[i]:b[i],c[i]:d[i]]
            frame_save=fits.PrimaryHDU(frame)
            frame_save_writeto(savepath_DX+str(names[i]))

    ##return the dictionary with the files names and the cut indexes:
    return dict_cut

##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

### as first thing, we run through all the frames, erase the 'BADROWS' ones, identify the star(s) position in each of them, and save all the info as
### a dictionary with a key for each frame name, together with the time of observation and the position of the star(s) in that frame:
os.chdir(path_to_frames)
all_frames=[f for f in sorted(os.listdir(path_to_frames)) if f.endswith('fits')]

if verbose==True:
    print ('STEP 1 : erasing the bad frames...')

##Step 1: erase all the bad frames:
bf_index=[] #empty array where to store the indeces of the bad frames
date_all=[] #empty array where to store the time of observation
all_arrays=[] #empty array where to store all the frames, datacube style
pbar=ProgressBar()
for i in pbar(range(len(all_frames))):
    data_i=fits.open(all_frames[i],memmap=False)
    hdu_i=data_i[0].header
    all_arrays.append(data_i[0].data)
    data_i.close()
    date_all.append(str(hdu_i['DATE-OBS'])+','+str(hdu_i['TIME-OBS']))
    try:
        bad=hdu_i['BADROWS']
        bf_index.append(i)
    except:
        pass

if verbose==True:
    print ('arrays saved')
    print (len(bf_index))
if len(bf_index)!=0:
    frames=all_frames #NOT removing the bad all_frames
    dates=date_all
    arrays=all_arrays
if len(bf_index)==0:
    frames=all_frames
    dates=date_all
    arrays=all_arrays

if verbose==True:
    print (len(bf_index),' bad frames have been erased. The analysis will proceed on ',len(frames),' frames.')


##Step 2: find the star(s) position in every frame:
###Let's parallelize this:

def parallel_starfinder(arrays):
    pool = multiprocessing.Pool()
    star_finder=partial(find_stars_parell,threshold=threshold,type='array',Bad_Pixel_map=Bad_Pixel_map,unsat=False)
    result_list=pool.map(star_finder,arrays)
    return result_list

if __name__ == '__main__':
    stars=parallel_starfinder(arrays)


dict_all=dict(zip(frames, zip(stars,dates,arrays))) #zip up the result in a dictionary where each key is a frame name, with attached time of observation and position of star(s) and frame.
####The time of obs is ALWAYS the index 1, and the element 0 can have different lenghts, depending on how many stars have been found in that frames, the array is ALWAYS stored in position 2
###check if the cut indexes 1 and 2 will be for which frame (DX or SX):
stars_pos=[dict_all[f][0] for f in sorted(dict_all.keys())]


if up_down==True:
    if stars_pos[50][0][1] < stars_pos[50][1][1]:
        print ('frame_index = DX_1')
        frame_index='DX_1'
    if stars_pos[50][0][1] > stars_pos[50][1][1]:
        print ('frame_index = DX_2')
        frame_index='DX_2'
if up_down==False:
    if stars_pos[50][0][0] > stars_pos[50][1][0]:
        print ('frame_index = DX_1')
        frame_index='DX_1'
    if stars_pos[50][0][0] < stars_pos[50][1][0]:
        print ('frame_index = DX_2')
        frame_index='DX_2'



##Step 3: consider only frames in which the specified number of stars have been found and separate the dictionary in 'up' and 'down' frames
#(or 'left' and 'right' depending on the specified nodding)
if verbose==True:
    print ('STEP 3 : separating the nods...')

new_dict={k: v for k, v in dict_all.items() if len(v[0]) == n_stars} ##considering only frames with the 'correct' number of n_stars
if up_down==True:
    nodA={k:v for k, v in new_dict.items() if v[0][0][0] > sep_ud}
    nodB={k:v for k, v in new_dict.items() if v[0][0][0] <= sep_ud}
if up_down==False:
    nodA={k:v for k, v in new_dict.items() if v[0][0][1] <= sep_lf}
    nodB={k:v for k, v in new_dict.items() if v[0][0][1] > sep_lf}
if verbose==True:
    print ('The frames have been divided in nodA and nodB frames')


###Now Let's apply the pca background subtraction:
if verbose==True:
    print ('STEP 4: Creating the sky frames for nodA and nodB...')

##nodA=sky and nodB=data
dict_cutB=pca_subtraction(master_flat,nodA,nodB,up_down,savepath_DX,savepath_SX,n_stars,Bad_Pixel_map,std=2)
#pca_subtraction(master_flat,nodA,nodB,up_down,savepath_DX,savepath_SX,n_stars,Bad_Pixel_map,std=2)
print ('nodA: DONE')
gc.collect()
##nodA=data and nodB=sky
dict_cutA=pca_subtraction(master_flat,nodB,nodA,up_down,savepath_DX,savepath_SX,n_stars,Bad_Pixel_map,std=2)
#pca_subtraction(master_flat,nodB,nodA,up_down,savepath_DX,savepath_SX,n_stars,Bad_Pixel_map,std=2)
gc.collect()
print ('nodB: DONE')
if verbose==True:
    print ('done: every frame has been sky subtracted with a pca approach and flat fielded using the master flat')


##Merge the dictionaries and save it into a pickle file:
dict_cut_all=dict_cutB
dict_cut_all.update(dict_cutA)
pickle.dump(dict_cut_all, open(savepath_DX+'../dict_cut_all.p', 'wb'))
gc.collect()


###add the pixel correction here, on the cut frames (so it does not have to correct the entire frame but only the cut stamps:
files_DX=[f for f in sorted(os.listdir(savepath_DX)) if f.endswith('.fits')]
files_SX=[f for f in sorted(os.listdir(savepath_SX)) if f.endswith('.fits')]
kernel=Gaussian2DKernel(x_stddev=2)

print ('Correcting the bad pixels:')
pbar=ProgressBar()
for i in pbar(range(len(files_DX))):
    DX_i=fits.open(savepath_DX+str(files_DX[i]),memmap=False)
    DX_i_array=DX_i[0].data
    DX_i.close()
    DX_i_corr=interpolate_replace_nans(DX_i_array,kernel)
    DX_i_corr_save=fits.PrimaryHDU(DX_i_corr)
    DX_i_corr_save.writeto(savepath_DX+str(files_DX[i]),overwrite=True)

gc.collect()
pbar=ProgressBar()
for i in pbar(range(len(files_SX))):
    SX_i=fits.open(savepath_SX+str(files_SX[i]),memmap=False)
    SX_i_array=SX_i[0].data
    SX_i.close()
    SX_i_corr=interpolate_replace_nans(SX_i_array,kernel)
    SX_i_corr_save=fits.PrimaryHDU(SX_i_corr)
    SX_i_corr_save.writeto(savepath_SX+str(files_SX[i]),overwrite=True)


gc.collect()

####################################################################################################
####################################################################################################

###STEP 6
###Now we subtract from each pixel the median of that column:
os.chdir(savepath_DX)
###To import the dictionary:
dict_all=pickle.load(open('../dict_cut_all.p', 'rb'))
files=[f for f in sorted(os.listdir(os.getcwd())) if f.endswith('.fits')]

datacube=[]
datacube_nan=[]
pbar=ProgressBar()
for i in pbar(range(len(files))):
    a1,b1,c1,d1,a2,b2,c2,d2=dict_all[str(files[i])]##get the cut indexes
    data_i,header_i=fits.getdata(savepath_DX+str(files[i]),0,header=True,memmap=False) ##get the data
    ###find the rough position of the star:
    try:
        pos_guess=[np.unravel_index(np.nanargmax(data_i[120:len(data_i)-120,120:len(data_i)-120]),np.shape(data_i[120:len(data_i)-120,120:len(data_i)-120]))[0] + 120,np.unravel_index(np.nanargmax(data_i[120:len(data_i)-120,120:len(data_i)-120]),np.shape(data_i[120:len(data_i)-120,120:len(data_i)-120]))[1]+120]
    except:
        print ('The fit did NOT work for frame '+str(files[i])+', the frame will be removed')
        os.remove(files[i])
        continue
    ###blank the pixels around the stars (i.e: = NaN):
    data_i_nan=np.copy(data_i)
    data_i_nan[pos_guess[0]-100:pos_guess[0]+100,pos_guess[1]-100:pos_guess[1]+100]=np.nan
    ##also: blank th pixels in a frame of 120 pixels around the border, just to exclude the possible presence of the other mirror:
    data_i_nan[0:50, :]=np.nan
    data_i_nan[:, 0:50]=np.nan
    data_i_nan[np.shape(data_i_nan)[0]-50:, :]=np.nan
    data_i_nan[:, np.shape(data_i_nan)[1]-50:]=np.nan
    ###pad the images to 2048x2048 and save both the one with nan and the one without:
    if frame_index=='DX_2':
        ##pad the image with Nans
        pad_r,pad_l,pad_u,pad_d=shape_data[1]-d2,c2,a2,(shape_data[0]-b2)
    if frame_index=='DX_1':
        ##pad the image with Nans:
        pad_r,pad_l,pad_u,pad_d=shape_data[1]-d1,c1,a1,(shape_data[0]-b1)
    ##append to the corresponding datacube:
    #import IPython
    #IPython.embed()
    gc.collect()
    datacube.append(np.pad(data_i,((pad_u,pad_d),(pad_l,pad_r)),mode='constant',constant_values=(np.nan,)))
    datacube_nan.append(np.pad(data_i_nan,((pad_u,pad_d),(pad_l,pad_r)),mode='constant',constant_values=(np.nan,)))


gc.collect()

save_datacube_nan=fits.PrimaryHDU(datacube_nan)
save_datacube_nan.writeto(savepath_DX+'../datacube_nan.fits')
save_datacube=fits.PrimaryHDU(datacube)
save_datacube.writeto(savepath_DX+'../datacube.fits')

datacube_nan=fits.open(savepath_DX+'../datacube_nan.fits')[0].data
datacube=fits.open(savepath_DX+'../datacube.fits')[0].data
datacube_median=np.reshape(np.repeat(np.nanmedian(datacube_nan,axis=1),np.shape(datacube)[1],axis=0),np.shape(datacube))
datacube_reduced=datacube-datacube_median

##Now save each median_subtracted fits file:
pbar=ProgressBar()
for i in pbar(range(len(datacube_reduced))):
    header_data_i=fits.open(savepath_DX+str(files[i]),memmap=False)
    hdu=header_data_i[0].header
    header_data_i.close()
    frame_i=fits.PrimaryHDU(datacube_reduced[i],header=hdu)
    frame_i.writeto(savepath_DX+str(files[i]),overwrite=True)

del datacube
del datacube_nan

##and same for SX:
###To import the dictionary:
os.chdir(savepath_SX)
dict_all=pickle.load(open('../dict_cut_all.p', 'rb'))
files=[f for f in sorted(os.listdir(os.getcwd())) if f.endswith('.fits')]

datacube=[]
datacube_nan=[]
pbar=ProgressBar()
for i in pbar(range(len(files))):
    a1,b1,c1,d1,a2,b2,c2,d2=dict_all[str(files[i])]##get the cut indexes
    data_i,header_i=fits.getdata(savepath_SX+str(files[i]),0,header=True,memmap=False) ##get the data
    ###find the rough position of the star:
    try:
        pos_guess=[np.unravel_index(np.nanargmax(data_i[120:len(data_i)-120,120:len(data_i)-120]),np.shape(data_i[120:len(data_i)-120,120:len(data_i)-120]))[0] + 120,np.unravel_index(np.nanargmax(data_i[120:len(data_i)-120,120:len(data_i)-120]),np.shape(data_i[120:len(data_i)-120,120:len(data_i)-120]))[1]+120]
    except:
        print ('The fit did NOT work for frame '+str(files[i])+', the frame will be removed')
        os.remove(files[i])
        continue
    ###blank the pixels around the stars (i.e: = NaN):
    data_i_nan=np.copy(data_i)
    data_i_nan[pos_guess[0]-100:pos_guess[0]+100,pos_guess[1]-100:pos_guess[1]+100]=np.nan
    ##also: blank th pixels in a frame of 120 pixels around the border, just to exclude the possible presence of the other mirror:
    data_i_nan[0:50, :]=np.nan
    data_i_nan[:, 0:50]=np.nan
    data_i_nan[np.shape(data_i_nan)[0]-50:, :]=np.nan
    data_i_nan[:, np.shape(data_i_nan)[1]-50:]=np.nan
    ###pad the images to 2048x2048 and save both the one with nan and the one without:
    if frame_index=='DX_1':
        ##pad the image with Nans
        pad_r,pad_l,pad_u,pad_d=shape_data[1]-d2,c2,a2,(shape_data[0]-b2)
    if frame_index=='DX_2':
        ##pad the image with Nans:
        pad_r,pad_l,pad_u,pad_d=shape_data[1]-d1,c1,a1,(shape_data[0]-b1)
    ##append to the corresponding datacube:
    gc.collect()
    datacube.append(np.pad(data_i,((pad_u,pad_d),(pad_l,pad_r)),mode='constant',constant_values=(np.nan,)))
    datacube_nan.append(np.pad(data_i_nan,((pad_u,pad_d),(pad_l,pad_r)),mode='constant',constant_values=(np.nan,)))

gc.collect()

save_datacube_nan=fits.PrimaryHDU(datacube_nan)
save_datacube_nan.writeto(savepath_SX+'../datacube_nan_SX.fits')
save_datacube=fits.PrimaryHDU(datacube)
save_datacube.writeto(savepath_SX+'../datacube_SX.fits')

datacube_nan=fits.open(savepath_DX+'../datacube_nan_SX.fits')[0].data
datacube=fits.open(savepath_DX+'../datacube_SX.fits')[0].data
datacube_median=np.reshape(np.repeat(np.nanmedian(datacube_nan,axis=1),np.shape(datacube)[1],axis=0),np.shape(datacube))
datacube_reduced=datacube-datacube_median

##Now save each fits file:
pbar=ProgressBar()
for i in pbar(range(len(datacube_reduced))):
    header_data_i=fits.open(savepath_SX+str(files[i]),memmap=False)
    hdu=header_data_i[0].header
    header_data_i.close()
    frame_i=fits.PrimaryHDU(datacube_reduced[i],header=hdu)
    frame_i.writeto(savepath_SX+str(files[i]),overwrite=True)

del datacube
del datacube_nan

print ('The data reduction is done: the frames have been sky-subtracted, flat-fielded and median subtracted (i.e: median of each column), with the star ROUGHLY positioned at the center of the frames. IF more than one mirror was used, the frames have been saved in DX and SX folder, separately. ')

##now let's do the fine centring:
exec(open("/Users/musso/Documents/Projects/Pyhton_scripts/fine_recentring.py").read())


end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))



##########################################################################################
##########################################################################################
if quick_reduction==False:
    if verbose==True:
        print ('/n The analysis is finished. The reduced images are saved together with the parallactic angles.')
if quick_reduction==True:
    print 'STEP 7: quick ADI reduction'
    classicalADI(savepath_DX_centre,savepath_DX_centre+'../',target_name+'_DX')
    classicalADI(savepath_SX_centre,savepath_SX_centre+'../',target_name+'_SX')
    print ('/n The analysis is finished. The reduced images are saved together with the parallactic angles and the classical reduce ADI images.')
