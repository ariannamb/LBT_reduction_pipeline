import numpy as np
from astropy.io import fits,ascii
import os
import scipy
from scipy.ndimage import interpolation
from progressbar import ProgressBar


##Classical ADI:
def classicalADI(folder,savefolder,name,save=True):
    #list the frames in the folder
    listfiles=[f for f in sorted(os.listdir(folder)) if f.endswith('.fits')]
    #prepare empty array where to store the angles
    angles=np.zeros((len(listfiles)))
    #get the size of a single frame (I assume squared frames)
    size=len(fits.open(folder+listfiles[0])[0].data)
    #empty array where to store all the frames
    array_all=np.zeros((len(listfiles),size,size))
    #Change directory
    os.chdir(folder)
    #loop over all frames and store the frame and the relative angle in the arrays
    pbar=ProgressBar()
    for i in range(len(array_all)):
        try:
            data_i=fits.open(listfiles[i])
            array_all[i]=data_i[0].data
            hdu_i=data_i[0].header
            angles[i]=hdu_i["NEW_PARA"]
            data_i.close()
        except:
            pass
    first_median=np.nanmedian(array_all, axis=0)
    array_all_final=np.zeros((len(listfiles),size,size))
    pbar2 = ProgressBar()
    for ii in range(len(array_all)):
        image_i=array_all[ii]
        image_sub=image_i - first_median
	##since interpolation does not work with arrays, I will substitute the Nans with a value:
        image_sub=np.nan_to_num(image_sub)
        image_rot=scipy.ndimage.interpolation.rotate(image_sub,-angles[ii], reshape=False)
        array_all_final[ii]=image_rot
    final_image=np.median(array_all_final,axis=0)
    if save==True:
        savefits=fits.PrimaryHDU(final_image)
        savefits.writeto(str(savefolder)+str(name)+'_NTH_ADI.fits',overwrite=True)
    return final_image ## the final image is north oriented

#classicalADI('/data/beegfs/astro-storage/groups/henning/musso/LIStEN_survey/Reduction/HNPeg/SCIENCE/SX_fcentre/','/data/beegfs/astro-storage/groups/henning/musso/LIStEN_survey/Reduction/HNPeg/SCIENCE/','HNPeg_SX')
#print 'done'
