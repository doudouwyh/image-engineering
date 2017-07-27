from PIL import Image
import numpy as np


im = Image.open('../pic/baboo256.BMP')

print im.mode,im.size

data = im.getdata()
data = np.array(data, 'float').reshape(im.size)
print data