import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('C:/Users/prani/Documents/AirSim/2021-11-14-15-55-57/images/*.png'):
    print("1")
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(size)
    img_array.append(img) 
print("2")
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()