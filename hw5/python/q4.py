import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions
    image = skimage.color.rgb2gray(skimage.restoration.denoise_bilateral(image, multichannel=True))
    thresh = skimage.filters.threshold_otsu(image)
    bw = skimage.morphology.closing(image < thresh, skimage.morphology.square(5))
    label_image = skimage.measure.label(bw, connectivity=2)
    regions = skimage.measure.regionprops(label_image)
    mean_area = np.mean([x.area for x in regions])
    for region in regions:
        if region.area >= mean_area / 2:
            bboxes.append(region.bbox)

    return bboxes, (~bw).astype(np.float)

# CFFFLKARMING
# DFFPERLEARAING
# CFCPE5TLEARNING
# T0D0LIST
# IMAXEAT0D0LIT
# ZCHFEK0FF7H8FIRFT
# 3HING0NT0D0LIST
# 3RZALIZEY0UHAVEALR6ADT
# C0MPLFTEDZTHINGI
# 9REWARDY0URSELFWITH
# ANAP
# 2BCDFFG
# HIIKLMN
# QPQRSTW
# VWXYZ
# 1Z3GS6789J
# HAIKUSAREGAGY
# BGTSZMETIMEGTHEYDQNTMAKGSGNGE
# RBFRIGBRAQR

