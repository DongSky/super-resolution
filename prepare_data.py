import os
from skimage.io import *
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
def prepare_data():
    DATA_PATH = "/Users/DongSky/Downloads/anime-faces/"
    HR_PATH = "/Users/DongSky/Downloads/faces/hr/"
    LR_PATH = "/Users/DongSky/Downloads/faces/lr/"
    paths = []
    for dirPath, dirNames, fileNames in os.walk(DATA_PATH):
        for filename in fileNames:
            p = os.path.join(dirPath, filename)
            if p.endswith(".jpg") or p.endswith(".png"):
                paths.append(p)
    for i in range(len(paths)):
        path = paths[i]
        hr_img = imread(path, as_grey=False)
        lr_img = resize(hr_img, (hr_img.shape[0] // 2, hr_img.shape[1] // 2))
        lr_img = resize(lr_img, (hr_img.shape[0], hr_img.shape[1]))
        if not os.path.exists(HR_PATH):
            os.makedirs(HR_PATH)
        if not os.path.exists(LR_PATH):
            os.makedirs(LR_PATH)
        lr = LR_PATH + str(i+1)+ ".jpg"
        hr = HR_PATH + str(i+1)+ ".jpg"
        imsave(fname=lr, arr=lr_img)
        imsave(fname=hr, arr=hr_img)
if __name__ == "__main__":
    prepare_data()
