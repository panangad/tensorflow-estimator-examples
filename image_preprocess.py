import glob2
import ntpath
import numpy
from PIL import Image

#Preprocess image file and extract float arrays
#First identifying mid point of each digits in the image and then extracting 7 x 9 pixel sorrounding that
def get_arrays_from_img(ifile):
    im = Image.open(ifile).convert('1') #Converting image to 8bit grey scale
    wi, he = im.size
    HEIGHT, WIDTH = he - 4, wi - 4
    crp = 2
    area = (crp,crp,wi - 2, he - 2)
    img = im.crop(area)    #Removing border from image
    dat = list(img.getdata())
    datt = list(map(lambda x: 1.0 if x < 100 else 0.0, dat))  #Converting to binary from grayscale 
    data = [datt[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]       #Converting to 2D array
    data_numpy = numpy.array(data)
    datatran = list(zip(*data))
    datatransum = list(map(lambda x: sum(x), datatran))
    st = False
    sidx,eidx = 0,0
    vs,ve = 100, 0
    numpos = []
    for idx, val in enumerate(datatransum):
        if(val > 0 and not st):
            sidx = idx
            st = True
        if(val < 1 and st):
            eidx = idx - 1
            st = False
            cx = (sidx+eidx)/2
            cy = (vs+ve)/2
            x1,x2,y1,y2 =int(cx-3.5),int(cx+3.5),int(cy-4.5),int(cy+4.5)
            numpos.append( data_numpy[:,range(x1,x2)][range(y1,y2+1),:].flatten() )
            vs,ve = 100, 0
        if(st):
            for vidx,vval in enumerate(datatran[idx]):
                if(vval > 0 and vs > vidx):
                    vs = vidx
                if(vval > 0 and ve < vidx):
                    ve = vidx
    return numpos