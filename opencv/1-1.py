# coding:utf-8
from PIL import Image
from numpy import *
from scipy.ndimage import *
import rof, harris
from pylab import *
import imtools, sift
pil_im = array(Image.open('img/empire.jpg').convert('L'))
# im = Image.open('img/AquaTermi_lowcontrast.JPG').convert('L')
# im2 = filters.gaussian_filter(im, 9)
#pil_im.show()
#Image.fromarray(im2).show()
# im = array(Image.open('img/houses.png').convert('L'))
# im = 1 * (im < 128)
# labels, nbr_objects = measurements.label(im)
# U,T = rof.denoise(pil_im, pil_im)
# figure()
# gray()
# imshow(pil_im)

# out = harris.compute_harris_response(pil_im)
# filtered_corrds = harris.get_harris_points(out, 6)
# harris.plot_harris_points(pil_im, filtered_corrds)
sift.process_image('/root/ss/ml/mlearn/opencv/img/empire.jpg', 'empire.sift')
# l1, d1 = sift.read_features_from_file('empire.sift')
# figure()
# gray()
# sift.plot_features(pil_im, l1, circle=True)
# show()









