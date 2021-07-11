import numpy as np
from PIL import Image
import glob


def square_crop_Image(img):
    w, h = img.size
    left = 0
    top = 0
    bottom = 0
    right = 0
    if w > h:
        diff = int((w-h)/2)
        left = diff
        right  = int(w - diff)
        bottom = h
    else:
        diff = int(h - w)/2
        right = w
        top = diff
        bottom = int(h-diff)

    sq_img = img.crop((left, top, right, bottom))
    return sq_img


images = []
for img in glob.glob("\Logistic Regression\Random\*.JPG"):
    img = Image.open(img)
    img = square_crop_Image(img)
    img = img.resize((500,500))
    img_arr = np.array(img)
    images.append(img_arr)

for i in range(len(images)):
    data = Image.fromarray(images[i])
    strng = str(i)+ "005.JPG"
    data.save(strng)





