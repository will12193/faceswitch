import os
import shutil
from imutils import paths
from pathlib import Path

topfolder = "./data/dataset2"

imagePaths=list(paths.list_images(topfolder))
for i in imagePaths:
    path = Path(i)
    parentPath = Path(path.parent.absolute())
    parentParentPath = str(parentPath.parent.absolute()) + "/" + i.split(os.path.sep)[-1]
    print(parentParentPath)
    shutil.move(str(i), parentParentPath)