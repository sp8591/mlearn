#coding: utf-8
from PIL import Image
from pylab import *
from scipy import ndimage
import numpy as np
from scipy.ndimage import filters, measurements
im  = array(Image.open('aa.png').convert('L'))
im2 = filters.gaussian_filter(im, 5)
imx = np.zeros(im.shape)
imy = np.zeros(im.shape)
filters.sobel(im, 1, imx)
filters.sobel(im, 0, imy)
im2 = 1 * (im < 128)
labels, nbr_objects = measurements.label(im)
print labels, nbr_objects
H = array([[1.4, 0.05, -100], [0.05, 1.4, -100], [0, 0, 1]])
im3 = ndimage.affine_transform(im, H[:2, :2], (H[0, 2], H[1, 2]))

imshow(im3)
#imshow(sqrt(imx**2 + imy**2))
figure()
imshow(imx)
print im.flatten(

)


figure()
hist(im.flatten(), 128)
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

plot([100, 100, 300], [200, 500, 400], 'cs')
plot([400, 400], [200, 500], 'mo-')
title('expire.jpg')
show()

def normalize(points):
    for row in points:
        row /= points[-1]
    return points

def make_homog(points):
    return vstack((points, ones((1, points.shape[1]))))

def H_from_points(fp, tp):
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    m = mean(fp[:2], axis=1)

import matplotlib._delaunay as md
x,y = array(np.random.standard_normal((2, 100)))
figure()
centers, edges, tri, neighbors = md.delaunay(x, y)
for t in tri:
    t_ext = [t[0], t[1], t[2], t[0]]
    plot(x[t_ext], y[t_ext], 'r')
plot(x,y, '*')
show()

