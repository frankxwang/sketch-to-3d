import numpy as np
import glob
import tables
import binvox_rw

ids = np.load("ids.npy")

fileh = tables.open_file('voxels.h5', mode='w')
atom = tables.Int8Atom()

voxels = fileh.create_earray(fileh.root, 'data', atom, (0, 128, 128, 128))

index = 0
for i in ids:
    with open("models-binvox/"+i+".binvox", 'rb') as f:
        model = binvox_rw.read_as_3d_array(f).data.astype(np.uint8)
    voxels.append([model])

    if index % 50 == 0:
        print(index)
    index += 1

print('Done!')
