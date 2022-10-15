import os
from pathlib import Path
import joblib

"""Transforms every parameter information file into a readable txt file."""

paths = Path('/home/vincent/machines').rglob('dict.txt')
for path in paths:
    dic = joblib.load(str(path))
    parent_dir = path.parent.absolute()
    with open(os.path.join(parent_dir, 'readable_dict.txt'), 'w') as f:
        f.write(str(parent_dir)+'\n\n---------PARAMS---------\n')
        for k, v in dic.items():
            if k == 'save_name':
                name = v
            f.write(str(k)+': '+str(v)+'\n')
        f.write('\n---------DATA PATHS---------\n')
        data_paths = Path('/home/vincent/data').rglob(name)
        for data_path in data_paths:
            f.write(str(data_path)+'\n')
    print(str(parent_dir))