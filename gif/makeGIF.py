import cv2
import numpy as np
import imageio

f1 = cv2.imread('0.jpg')
f2 = cv2.imread('1.jpg')
f3 = cv2.imread('2.jpg')
f4 = cv2.imread('3.jpg')
frames = [f1, f2, f3, f4]
lis = []
for i in range(len(frames)):
    area = frames[i].shape[0]*frames[i].shape[1]
    lis.append(area)

ind = np.argmin(lis)
height = frames[ind].shape[0]
width = frames[ind].shape[1]
dim = (width, height)
# print((height, width))

# print("Before resizing")
# for i in frames:
#     print(i.shape)

for i in range(len(frames)):
    frames[i] = cv2.resize(frames[i], dim, interpolation = cv2.INTER_AREA)
    frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)


# print("After resizing")
# for i in frames:
#     print(i.shape)
imageio.mimsave('result.gif', frames, fps=4)
print("Done!")