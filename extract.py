import numpy as np
import tables
import glob
import cv2
# from matplotlib import pyplot as plt

fileh = tables.open_file('sketches.h5', mode='w')
atom = tables.Int8Atom()

images = fileh.create_earray(fileh.root, 'data', atom, (0, 8, 512, 512))

# ids = []
ids = np.load("ids.npy")
index = 0
for i in ids:
    i = 'screenshots/'+i+'/'
    obj = []
    for j in glob.glob(i+'*.png'):
        a, b = j.split('-')
        if int(b[:-4]) >= 6:
            image = cv2.imread(j, 0)
            edges = cv2.Canny(image, 100, 200)
            edges = np.divide(edges, 255)
            obj.append(edges)

    # name = i.split('/')[-2]
    # ids.append(name)
    images.append([obj])

    if index % 50 == 0:
        print(index)
    index += 1

# print("Saving images")
# np.save('tensors', np.array(tensor))
# print("Saving ids")
# np.save('ids', np.array(ids))
print('Done!')
