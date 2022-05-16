import os
import shutil
from imutils import paths
from pathlib import Path

topfolder = "./data/dataset2.1/without_mask"

imagePaths=list(paths.list_images(topfolder))
delete = True
for i in imagePaths:
    # path = Path(i)
    # parentPath = Path(path.parent.absolute())
    # parentParentPath = str(parentPath.parent.absolute()) + "/" + i.split(os.path.sep)[-1]
    # print(parentParentPath)
    # shutil.move(str(i), parentParentPath)
    # if delete:
    #     os.remove(i)
    #     delete = False
    # else:
    #     delete = True