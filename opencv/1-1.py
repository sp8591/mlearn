# coding:utf-8
from PIL import Image
from numpy import *
from scipy.ndimage import *
import rof, harris, warp, sift
from pylab import *
import imtools, sift, ransac

im1 = array(Image.open('img/empire.jpg').convert('L'))
im2 = array(Image.open('img/crans_1_small.jpg').convert('L'))
im3 = array(Image.open('img/crans_2_small.jpg').convert('L'))
sift.process_image('img/crans_1_small.jpg', 'crans_1_small.sift')
sift.process_image('img/crans_2_small.jpg', 'crans_2_small.sift')
l2, d2 = sift.read_features_from_file('crans_1_small.sift')
l3, d3 = sift.read_features_from_file('crans_2_small.sift')
match = sift.match(d2, d3)
figure()
gray()


sift.plot_features(im2, l2)
figure()
gray()
sift.plot_features(im3, l3)

figure()
harris.plot_matches(im2, im3, l2, l3, match)
show()

#ransac.test()
# featname = ['Univ' + str(i+1) + '.sift' for i in range(5)]
# imname = ['Univ' + str(i+1) + '.jpg' for i in range(5)]
# l = {}
# d = {}
# for i in range(5):
#     #sift.process_image('/root/ss/ml/mlearn/opencv/img/' + imname[i], featname[i])
#     l[i], d[i] = sift.read_features_from_file(featname[i])
# matches = {}
# for i in range(4):
#     matches[i] = sift.match(d[i], d[i+1])
#     figure()
#     gray()
#     sift.plot_matches(array(Image.open('/root/ss/ml/mlearn/opencv/img/' + imname[i])), array(Image.open('/root/ss/ml/mlearn/opencv/img/' + imname[i+1])),
#                       l[i], l[i+1], matches[i], False)
show()


# gray()
# imshow(im2)
# p = array(ginput(4)).T
# p = concatenate((p, [[1, 1, 1, 1]]), axis=0).astype('int')
# print p
# p[[0, 1], :] = p[[1, 0], :]
# #p = array([[519,704,42,688],[264,266,434,440],[1,1,1,1]])
#
# im3 = warp.image_in_image(im1, im2, p)
# figure()
# gray()
# imshow(im3)
# show()
#
# print p

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
# sift.process_image('/root/ss/ml/mlearn/opencv/img/empire.jpg', 'empire.sift')
# l1, d1 = sift.read_features_from_file('empire.sift')
# figure()
# gray()
# sift.plot_features(pil_im, l1, circle=True)
# show()









