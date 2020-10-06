
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import numpy as np
import scipy
from numba import prange
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from datetime import datetime
from progressbar import ProgressBar
import datetime
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import sys
import resource
from multiprocessing import Pool
from photutils import centroid_com, centroid_1dg, centroid_2dg
import photutils

#sys.path.append('/Users/musso/Documents/Projects/NaCo-packages/Pipeline_modules') ##Not needed in astronode4
#from side_functions import angular_coords_float,planets_finder,twoD_Gaussian


def find_stars(path,threshold,type,Bad_Pixel_map,unsat=False):
    #######
    #import the image as an array
    if type=='fits':
        image=fits.open(path)
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
	#import IPython
	#IPython.embed()
        return np.array([])
#######

def skies(dictionary,diff,nod):
    keys=sorted(dictionary.keys())
    skies=[]
    t_mean=[]
    empty=[]
    t_empty=[]
    names=[]
    count=0
    for i in range(1,len(keys)):
        try:
            time_ii=datetime.datetime.strptime(dictionary[keys[i-1]][1],'%Y-%m-%d,%H:%M:%S.%f') #convert the date-time string in an actual datetime object
            time_i=datetime.datetime.strptime(dictionary[keys[i]][1],'%Y-%m-%d,%H:%M:%S.%f')
        except:
            time_ii=datetime.datetime.strptime(dictionary[keys[i-1]][1],'%Y%m%d,%H%M%S') #convert the date-time string in an actual datetime object
            time_i=datetime.datetime.strptime(dictionary[keys[i]][1],'%Y%m%d,%H%M%S')
        value=(time_i-time_ii).total_seconds() #evaluate the difference in seconds
        if i == (len(keys) -1 ):
            empty.append(dictionary[keys[i]][2])
            t_empty.append(time_i)
            t_mean.append(t_empty[0] +datetime.timedelta(seconds=(t_empty[0]-t_empty[-1]).total_seconds()))
            skies.append(np.median(np.array(empty),axis=0))
            names.append(str(nod)+'_sky_'+str(count)+'.fits')
            print('DONE')
        if value <= diff:
            empty.append(dictionary[keys[i-1]][2])
            t_empty.append(time_ii)
        if value > diff:
            empty.append(dictionary[keys[i-1]][2])
            t_empty.append(time_ii)
            t_mean.append(t_empty[0] +datetime.timedelta(seconds=(t_empty[0]-t_empty[-1]).total_seconds()))
            skies.append(np.median(np.array(empty),axis=0))
            names.append(str(nod)+'_sky_'+str(count)+'.fits')
            count=count+1
            empty=[]
            t_empty=[]
    return dict(zip(names,zip(t_mean,skies)))

####

def nodding(flat,nod,sky,up_down,savepath_DX,savepath_SX,n_stars,Bad_Pixel_map,std=2):
    pbar=ProgressBar()
    ##save all the rough_sizes in an array:
    sizes=[]
    for g in pbar(prange(len(nod))):
        frame_g=nod[sorted(nod.keys())[g]][2]#get the frame
        time_g=nod[sorted(nod.keys())[g]][1] #extract the observation time for this frames
        try:
            time_g=datetime.datetime.strptime(time_g,'%Y-%m-%d,%H:%M:%S.%f') #convert it to a proper datetime object
        except:
            time_g=datetime.datetime.strptime(time_g,'%Y%m%d,%H%M%S') #convert it to a proper datetime object
	#print time_g
    times_b=[val[0] for key , val in sorted(sky.items()) ] #get all the average times for skies
    skies_name=[key for key , val in sorted(sky.items()) ] #get all the names of the skies
    diff_ts=[]
	#import IPython
	#IPython.embed()
    for gg in range(len(times_b)):
        diff_time= np.abs((time_g - times_b[gg] ).total_seconds())
        diff_ts.append(diff_time)
    sky_g=sky[skies_name[np.argmin(diff_ts)]][1]
    reduced_frame_g=np.divide((np.asarray(frame_g,dtype='float64')  - np.asarray(sky_g,dtype='float64')), flat) # frame = (raw-sky)/flat
    ###And now let's take care of the bad pixels:
    mask=np.isnan(Bad_Pixel_map) ##get the mask
    kernel=Gaussian2DKernel(stddev=std) ##generate the gaussian kernel with the desired standard deviation
    reduced_frame_g=np.array(reduced_frame_g,dtype=float)
    reduced_frame_g[mask]=np.nan #set the bad pixels to NaN value
    reduced_frame_g=interpolate_replace_nans(reduced_frame_g, kernel) ##correct the bad pixels
	##now, using the separation information, let's cut the reduced frame around the star, with the desired size (allowing extra size for re-centring):
    ##The frames will be cut using half the distance between the x coordinates (or y coordinates)) of the stars:
    stars_pos=np.array(nod[sorted(nod.keys())[g]][0]) #store the positions of the two stars in a more manageable array
    if n_stars>1:
        if up_down==True:
            rough_size=np.int((np.abs(stars_pos[0][1] - stars_pos[1][1])) - 20) #evaluate the rough size of the stamps (as the distance between the two stars, minus ~60 pixels to exclude the other star)
        if up_down==False:
            rough_size=np.int((np.abs(stars_pos[0][0] - stars_pos[1][0])) - 20) #evaluate the rough size of the stamps (as the distance between the two stars, minus ~60 pixels to exclude the other star)
    if n_stars==1:
        rough_size=np.int((np.min([len(frame_g[1])-stars_pos[0][1],len(frame_g[0])-stars_pos[0][0]]))/2. -22)
    if rough_size % 2 ==0:
        rough_size=rough_size - 0 # the final size must be an even number
    if rough_size % 2 !=0:
        rough_size=rough_size - 1 # the final size must be an even number
    if n_stars>1:
        a1=np.int(stars_pos[0][0]) - np.int(rough_size)
        b1=np.int(stars_pos[0][0]) + np.int(rough_size)
        c1=np.int(stars_pos[0][1]) - np.int(rough_size)
        d1=np.int(stars_pos[0][1]) + np.int(rough_size)
        a2=np.int(stars_pos[1][0]) - np.int(rough_size)
        b2=np.int(stars_pos[1][0]) + np.int(rough_size)
        c2=np.int(stars_pos[1][1]) - np.int(rough_size)
        d2=np.int(stars_pos[1][1]) + np.int(rough_size)
        if a1<0:
            a1=0
        if c1<0:
            c1=0
        if a2<0:
            a2=0
        if c2<0:
            c2=0
        ###create two zeros frames:
        frame_1=np.full(np.shape(frame_g),np.nan)
        frame_2=np.full(np.shape(frame_g),np.nan)
        frame_1[a1:b1,c1:d1]=reduced_frame_g[a1:b1,c1:d1]
        frame_2[a2:b2,c2:d2]=reduced_frame_g[a2:b2,c2:d2]
#	    if np.shape(frame_1)[0]==0 or np.shape(frame_1)[1]==0 or np.shape(frame_2)[0]==0 or np.shape(frame_2)[1]==0:
        #now save the two stamps, by default the DX mirror is either the leftmost (for up_down nodding) or the uppermost (for left-right nodding):
        if up_down==True:
            if stars_pos[0][1] < stars_pos[1][1]:
                frame_DX=frame_1
                frame_SX=frame_2
            if stars_pos[0][1] > stars_pos[1][1]:
                frame_DX=frame_2
                frame_SX=frame_1
        if up_down==False:
            if stars_pos[0][0] > stars_pos[1][0]:
                frame_DX=frame_1
                frame_SX=frame_2
            if stars_pos[0][0] < stars_pos[1][0]:
                frame_DX=frame_2
                frame_SX=frame_1
            ##save:
        save_DX=fits.PrimaryHDU(frame_DX)
        save_DX.writeto(savepath_DX + 'DX_'+str(sorted(nod.keys())[g]),overwrite=True)
        save_SX=fits.PrimaryHDU(frame_SX)
        save_SX.writeto(savepath_SX + 'SX_'+str(sorted(nod.keys())[g]),overwrite=True)
    if n_stars==1:
	    #print 'rough_size= '+str(rough_size)
	    #import IPython
	    #IPython.embed()
	    ##check that the stamp is not bigger than the frame:
        a=np.int(stars_pos[0][0]) - np.int(rough_size)
        b=np.int(stars_pos[0][0]) + np.int(rough_size)
        c=np.int(stars_pos[0][1]) - np.int(rough_size)
        d=np.int(stars_pos[0][1]) + np.int(rough_size)
        if a<0:
            a=0
        if c<0:
            c=0
	    ###create two zeros frames:
        frame_1=np.full(np.shape(frame_g),np.nan)
        frame_1[a:b,c:d]=reduced_frame_g[a:b,c:d]
        if np.shape(frame_1)[0]==0 or np.shape(frame_1)[1]==0:
            save=fits.PrimaryHDU(frame_1)
            save.writeto(savepath_DX + 'DX_'+str(sorted(nod.keys())[g]),overwrite=True)
    sizes.append(rough_size)
    #return sizes, np.min(np.array(sizes))

#####
def cut_and_centre(folder,root_folder,new_size,filetype='files'):
    pbar = ProgressBar()
    #set the directories with the original frames where to get the header, and the reduced frames:
    dir_files=folder
    dir_para=root_folder
    #set the radius to fit the gaussian:
    radius= 3
    para_all=[] #empty array where to store all the parallactic angles
    erased_frames=0 # count to see for how many frames the fit fails and I erase them
    #set the current directory and make a list of all the fits files in there:
    files=[f for f in sorted(os.listdir(dir_files)) if f.endswith('.fits')]
    #loop over all files, open the fits, find the real center, cut, and save with the correct header
    os.chdir(dir_files)
    dict_centring=dict() ##empty dictionary where to store the info about where the frame was cut
    for i in pbar(prange(len(files))):
        ##Open the fits file
        data_i=fits.open(files[i],memmap=False)
        data=data_i[0].data
        data_i.close()
        size=len(data)
        #pos_guess=[size/2.,size/2.]
        try:
	    ##To avoid the second mirror contamintaing this (i.e: the second star might be visible in the border of the frame) I will look for the maximum value of the array excluding a stripe of 120 pixels around the border:
            pos_guess=[np.unravel_index(np.nanargmax(data[120:len(data)-120,120:len(data)-120]),np.shape(data[120:len(data)-120,120:len(data)-120]))[0] + 120,np.unravel_index(np.nanargmax(data[120:len(data)-120,120:len(data)-120]),np.shape(data[120:len(data)-120,120:len(data)-120]))[1]+120]
        except:
            print('The fit did NOT work for frame '+str(files[i])+', the frame will be removed')
            os.remove(files[i])
            erased_frames=erased_frames+1
            continue
	#open and get the correct header for this file:
        #get random file in the dir para folder to extract the root
        root_name=(os.listdir(dir_para)[20]).rsplit('_',1)[0]
        #get the filenumber of this current file:
        file_num=files[i][-(len(os.listdir(dir_para)[20])-len(root_name)):]
        #get the relevant header
        header_i=fits.open(dir_para+str(root_name)+str(file_num))
        header=header_i[0].header
        header_i.close()
        #add the NEW_PARA card to the header:
        angle=header["LBT_PARA"]
        #para_all.append(angle)
        header["NEW_PARA"]=angle
	### now let's roughly cut the frame around the brightest pixel:
        new_frame=data[pos_guess[0] - new_size/2 : pos_guess[0] + new_size/2,pos_guess[1] - new_size/2 : pos_guess[1] + new_size/2]
        #save the new frame as a fits file, with the correct header, subscribing the existent file:
        ###first check if the frame as a weird size:
        try:
            if np.shape(np.array((new_frame)))==(new_size,new_size):
                ##convert all Nan to zeros:
                new_frame=np.nan_to_num(new_frame)
                save_frame=fits.PrimaryHDU(new_frame, header=header)
                save_frame.writeto(dir_files+str(files[i]),overwrite=True)
                para_all.append(angle)
                dict_centring.update({str(files[i]):((pos_guess[0] - new_size/2,pos_guess[0] + new_size/2,pos_guess[1] - new_size/2,pos_guess[1] + new_size/2))})
            if np.shape(np.array((new_frame)))!=(new_size,new_size):
                print( 'The fit did NOT work for frame '+str(files[i]))
                os.remove(files[i])
                erased_frames=erased_frames+1
        except:
            print ('The fit did NOT work for frame '+str(files[i]))
            os.remove(files[i])
            erased_frames=erased_frames+1
            continue
    print ('The fit did not work for ',erased_frames,' frames.')
    #import IPython
    #IPython.embed()
    ##Return the parallactic angles as an array and the info about how the frames where cut:
    return np.array(para_all), dict_centring

def para_angles(folder,root_folder):
    pbar = ProgressBar()
    #set the directories with the original frames where to get the header, and the reduced frames:
    dir_files=folder
    dir_para=root_folder
    #set the current directory and make a list of all the fits files in there:
    files=[f for f in sorted(os.listdir(dir_files)) if f.endswith('.fits')]
    #loop over all files, open the fits, find the real center, cut, and save with the correct header
    para_all=[] #empty array where to store all the parallactic angles
    os.chdir(dir_files)
    datacube=[]
    for i in pbar(prange(len(files))):
        ##Open the fits file
        data_i=fits.open(files[i])
        data=data_i[0].data
        data_i.close()
        datacube.append(data)
        ##open and get the correct header for this file:
        #get random file in the dir para folder to extract the root
        root_name=(os.listdir(dir_para)[20]).rsplit('_',1)[0]
        #get the filenumber of this current file:
        file_num=files[i][-(len(os.listdir(dir_para)[20])-len(root_name)):]
        #get the relevant header
        header_i=fits.open(dir_para+str(root_name)+str(file_num))
        header=header_i[0].header
        header_i.close()
        #extract the  NEW_PARA card from the header:
        angle=header["LBT_PARA"]
        para_all.append(angle)
        header["NEW_PARA"]=angle
        save_frame=fits.PrimaryHDU(data, header=header)
        save_frame.writeto(dir_files+str(files[i]),overwrite=True)
    ##Return the parallactic angles as an array:
    return np.array(para_all),np.array(datacube)
