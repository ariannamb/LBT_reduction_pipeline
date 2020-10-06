import os
from astropy.io import fits
import numpy as np
from photutils import centroid_com, centroid_1dg, centroid_2dg
from progressbar import ProgressBar
import sys
from classicalADI import *

pbar=ProgressBar()

###set the root folder for the extraction of the header and parallactic angles:
dir_para='/Users/musso/Documents/Projects/Pyhton_scripts/HD143894_data/raw/SCIENCE/'

##set the folder with the frames that need recentring:
folder_DX='/Users/musso/Documents/Projects/Pyhton_scripts/HD143894_data/reduced/SCIENCE/DX/'
#set the flder where to save the re-centred files:
save_folder_DX='/Users/musso/Documents/Projects/Pyhton_scripts/HD143894_data/reduced/SCIENCE/DX_fcentre/'
#set a counter that takes care of how many frames fail to re-centre:
erased_frames=0

###set the folder with the frames that need recentring:
folder_SX='/Users/musso/Documents/Projects/Pyhton_scripts/HD143894_data/reduced/SCIENCE/SX/'
##set the flder where to save the re-centred files:
save_folder_SX='/Users/musso/Documents/Projects/Pyhton_scripts/HD143894_data/reduced/SCIENCE/SX_fcentre/'


size_new=400 ##final size for the frames

############################################################################################################
############################################################################################################
os.chdir(folder_DX)
#para_angles_DX=[]

#get random file in the dir para folder to extract the root
root_name=(os.listdir(dir_para)[20]).rsplit('_',1)[0]

##get all the files in the folder:
files_DX=[f for f in sorted(os.listdir(folder_DX)) if f.endswith('.fits')]
for i in pbar(range(len(files_DX))):
    data_i=fits.open(files_DX[i],memmap=False)
    data=data_i[0].data
    hdu=data_i[0].header
    data_i.close()
    ###find the rough position of the star:
    pos_guess=np.unravel_index(np.argmax(data), np.shape(data))
    ###check the minimum distance between the star and the edges of the frame, and roughly cut a square around the star:
    rough_size=size_new+100
    new_frame=data[int(pos_guess[0]) - int(rough_size/2 ): int(pos_guess[0]) + int(rough_size/2),int(pos_guess[1]) - int(rough_size/2):int( pos_guess[1] )+ int(rough_size/2)]
    #save_new=fits.PrimaryHDU(new_frame, header=hdu)
    #save_new.writeto('/Users/musso/Documents/Projects/Pyhton_scripts/HD143894_data/reduced/SCIENCE/DX_new_frame/'+str(files_DX[i]),overwrite=True)
    #import IPython
    #IPython.embed()
    ###now perform the finer centering on the roughly cut frame:
    #erased_frames=erased_frames+1
    ##try to fit:
    try:
        y,x=centroid_1dg(new_frame)
        #print(files_DX[i])
        #print(x,y)
        final_frame=new_frame[np.int(x)-int(size_new/2):np.int(x)+int(size_new/2),np.int(y)-int(size_new/2):np.int(y)+int(size_new/2)]
        if np.shape(final_frame)==(size_new,size_new):
            #get the filenumber of this current file:
            file_num=files_DX[i][-(len(os.listdir(dir_para)[20])-len(root_name)):]
            #get the relevant header
            header_i=fits.open(dir_para+str(root_name)+str(file_num),memmap=False)
            header=header_i[0].header
            header_i.close()
            #extract the  NEW_PARA card from the header:
            angle=header["LBT_PARA"]
            header["NEW_PARA"]=angle
            #para_angles.append(angle)
            save_frame=fits.PrimaryHDU(final_frame, header=header)
            save_frame.writeto(str(save_folder_DX)+str(files_DX[i]),overwrite=True)
        else:
           print ('frame too small '+str(files_DX[i]))
           erased_frames=erased_frames+1
           continue
    except:
        print ('fit did not work for frame '+str(files_DX[i]))
        erased_frames=erased_frames+1
        continue

print ('The fine recentring of the DX files is done. The re-centring did not work for '+str(erased_frames)+'.')

##save the parallactic angles:
#para_angles_DX_save=fits.PrimaryHDU(np.array(para_angles_DX))
#para_angles_DX_save.writeto(save_folder_DX+'../parang_DX.fits',clobber=1)

############################################################################################################
############################################################################################################


pbar=ProgressBar()

erased_frames=0
os.chdir(folder_SX)
#para_angles_SX=[]

##get all the files in the folder:
files_SX=[f for f in sorted(os.listdir(folder_SX)) if f.endswith('.fits')]
for i in pbar(range(len(files_SX))):
    data_i=fits.open(files_SX[i],memmap=False)
    data=data_i[0].data
    hdu=data_i[0].header
    data_i.close()
    ###find the rough position of the star:
    pos_guess=np.unravel_index(np.argmax(data), np.shape(data))
    ###check the minimum distance between the star and the edges of the frame, and roughly cut a square around the star:
    rough_size=size_new+50
    new_frame=data[int(pos_guess[0]) - int(rough_size/2 ):int( pos_guess[0]) + int(rough_size/2),int(pos_guess[1]) - int(rough_size/2):int( pos_guess[1]) +int( rough_size/2)]
    #import IPython
    #IPython.embed()
    ###now perform the finer centering on the roughly cut frame:
    #erased_frames=erased_frames+1
    ##try to fit:
    try:
        y,x=centroid_1dg(new_frame)
        #print files[i]
        #print x,y
        final_frame=new_frame[int(np.int(x)-size_new/2):int(np.int(x)+size_new/2),int(np.int(y)-size_new/2):int(np.int(y)+size_new/2)]
        if np.shape(final_frame)==(size_new,size_new):
            #get the filenumber of this current file:
            file_num=files_SX[i][-(len(os.listdir(dir_para)[20])-len(root_name)):]
            #get the relevant header
            header_i=fits.open(dir_para+str(root_name)+str(file_num),memmap=False)
            header=header_i[0].header
            header_i.close()
            #extract the  NEW_PARA card from the header:
            angle=header["LBT_PARA"]
            header["NEW_PARA"]=angle
            #para_angles.append(angle)
            save_frame=fits.PrimaryHDU(final_frame, header=header)
            save_frame.writeto(str(save_folder_SX)+str(files_SX[i]),overwrite=True)
        else:
           print ('frame too small '+str(files_SX[i]))
           erased_frames=erased_frames+1
           continue
    except:
        print ('fit did not work for frame '+str(files_SX[i]))
        erased_frames=erased_frames+1
        continue


print ('The fine recentring of the SX files is done. The re-centring did not work for '+str(erased_frames)+'.')

##run classicalADI:
classicalADI(save_folder_DX,save_folder_DX+'../','HD143894_DX')
classicalADI(save_folder_SX,save_folder_SX+'../','HD143894_SX')
#print '/n The analysis is finished. The reduced images are saved together with the parallactic angles and the classical reduce ADI images.'
