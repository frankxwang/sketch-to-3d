import glob
import numpy as np

visited = {}

ids = np.array([])

for i in glob.glob('screenshots/*/'):
    id = i.split('/')[-2]
    visited[id] = True

for i in glob.glob('models-binvox/*.binvox'):
    id = i.split("/")[-1][:-7]
    if id in visited:
        ids = np.append(ids, id)

print(ids)
np.save("ids", ids)
