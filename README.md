# LBT_reduction_pipeline
End-to-end data reduction pipeline for LBT/LMIRcam data taken in dual imaging mode

This pipeline has been created to reduce LBT data taken with the LMIRCam instrument in dual imaging mode, using the L' band filter at ~3.8 micron, specifically for the L' band Imaging Survey for Exoplanets in the North (LIStEN).
An overview of the pipeline can be found in the paper presenting the LIStEN survey (Musso Barcucci et al., 2020, submitted to A&A).

The pipeline consists of the following steps:
-Data organisation
-Locating the star(s) position
-Nod separation
-Sky subtraction
-Bad pixel correction
-Bad stripes correction
-De-warping
-Centring
-Stacking
