import glob

import PIL
import cv2
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


class record:
    def __init__(self,**kw): self.__dict__.update(kw)


def disp_img(img, title, h, w):
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, (w, h))


def isintarray(a):
    return a.dtype in [np.dtype('B'),np.dtype('int16'),np.dtype('int32'),np.dtype('int64'),
                       np.dtype('uint16'),np.dtype('uint32'),np.dtype('uint64')]

def isintegerarray(a):
    return a.dtype in [np.dtype('int32'),np.dtype('int64'),np.dtype('uint32'),np.dtype('uint64')]


def dim0(s):
    """Dimension of the slice list for dimension 0."""
    return s[0].stop-s[0].start


def dim1(s):
    """Dimension of the slice list for dimension 1."""
    return s[1].stop-s[1].start


def norm_max(a):
    return a/np.amax(a)


def width(s):
    return s[1].stop-s[1].start


def area(a):
    """Return the area of the slice list (ignores anything past a[:2]."""
    return np.prod([max(x.stop - x.start, 0) for x in a[:2]])

def center(s):
    ycenter = np.mean([s[0].stop,s[0].start])
    xcenter = np.mean([s[1].stop,s[1].start])
    return (ycenter, xcenter)


def r_dilation(image,size,origin=0):
    """Dilation with rectangular structuring element using maximum_filter"""
    return ndimage.maximum_filter(image,size,origin=origin)


def r_erosion(image,size,origin=0):
    """Erosion with rectangular structuring element using maximum_filter"""
    return ndimage.minimum_filter(image,size,origin=origin)


# def r_opening(image,size,origin=0):
#     """Opening with rectangular structuring element using maximum/minimum filter"""
#     check_binary(image)
#     image = r_erosion(image,size,origin=origin)
#     return r_dilation(image,size,origin=origin)
#
#
# def r_closing(image,size,origin=0):
#     """Closing with rectangular structuring element using maximum/minimum filter"""
#     check_binary(image)
#     image = r_dilation(image,size,origin=0)
#     return r_erosion(image,size,origin=0)


def rb_dilation(image,size,origin=0):
    """Binary dilation using linear filters."""
    output = np.zeros(image.shape,'f')
    ndimage.uniform_filter(image,size,output=output,origin=origin,mode='constant',cval=0)
    return np.array(output>0,'i')


def rb_erosion(image,size,origin=0):
    """Binary erosion using linear filters."""
    output = np.zeros(image.shape,'f')
    ndimage.uniform_filter(image,size,output=output,origin=origin,mode='constant',cval=1)
    return np.array(output==1,'i')


def rb_opening(image,size,origin=0):
    """Binary opening using linear filters."""
    image = rb_erosion(image,size,origin=origin)
    return rb_dilation(image,size,origin=origin)


# def rb_closing(image,size,origin=0):
#     """Binary closing using linear filters."""
#     image = rb_dilation(image,size,origin=origin)
#     return rb_erosion(image,size,origin=origin)


def select_regions(binary,f,min=0,nbest=100000):
    """Given a scoring function f over slice tuples (as returned by
    find_objects), keeps at most nbest regions whose scores is higher
    than min."""
    labels,n = ndimage.label(binary)
    objects = ndimage.find_objects(labels)
    scores = [f(o) for o in objects]
    best = np.argsort(scores)
    keep = np.zeros(len(objects)+1,'i')
    if nbest > 0:
        for i in best[-nbest:]:
            if scores[i]<=min: continue
            keep[i+1] = 1
    return keep[labels]


def check_binary(image):
    assert image.dtype=='B' or image.dtype=='i' or image.dtype==np.dtype('bool'),\
        "array should be binary, is %s %s"%(image.dtype,image.shape)
    assert np.amin(image)>=0 and np.amax(image)<=1,\
        "array should be binary, has values %g to %g"%(np.amin(image),np.amax(image))


def read_image_binary(fname, dtype='i'):
    """Read an image from disk and return it as a binary image
    of the given dtype."""
    pil = PIL.Image.open(fname)
    a = pil2array(pil)
    if a.ndim == 3: a = np.amax(a, axis=2)
    return np.array(a > 0.5 * (np.amin(a) + np.amax(a)), dtype)


def pil2array(im, alpha=0):
    if im.mode == "L":
        a = np.fromstring(im.tobytes(), 'B')
        a.shape = im.size[1], im.size[0]
        return a
    if im.mode == "RGB":
        a = np.fromstring(im.tobytes(), 'B')
        a.shape = im.size[1], im.size[0], 3
        return a
    if im.mode == "RGBA":
        a = np.fromstring(im.tobytes(), 'B')
        a.shape = im.size[1], im.size[0], 4
        if not alpha: a = a[:, :, :3]
        return a
    return pil2array(im.convert("L"))


def correspondences(labels1,labels2):
    """Given two labeled images, compute an array giving the correspondences
    between labels in the two images."""
    q = 100000
    assert np.amin(labels1)>=0 and np.amin(labels2)>=0
    assert np.amax(labels2)<q
    combo = labels1*q+labels2
    result = np.unique(combo)
    result = np.array([result//q,result%q])
    return result


def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res


def propagate_labels(image,labels,conflict=0):
    """Given an image and a set of labels, apply the labels
    to all the regions in the image that overlap a label.
    Assign the value `conflict` to any labels that have a conflict."""
    rlabels,_ = ndimage.label(image)
    cors = correspondences(rlabels,labels)
    outputs = np.zeros(np.amax(rlabels)+1,'i')
    oops = -(1<<30)
    for o,i in cors.T:
        if outputs[o]!=0: outputs[o] = oops
        else: outputs[o] = i
    outputs[outputs==oops] = conflict
    outputs[0] = 0
    return outputs[rlabels]


def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background"""
    distances,features = ndimage.distance_transform_edt(labels==0,return_distances=True,return_indices=True)
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances<maxdist)
    return spread


def compute_lines(segmentation,scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = ndimage.find_objects(segmentation)
    lines = []
    for i,o in enumerate(lobjects):
        if o is None: continue
        if dim1(o)<2*scale or dim0(o)<scale: continue
        mask = (segmentation[o]==i+1)
        if np.amax(mask)==0: continue
        result = record()
        result.label = i+1
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines


def reading_order(lines, highlight=None):
    """Given the list of lines (a list of 2D slices), computes
    the partial reading order.  The output is a binary 2D array
    such that order[i,j] is true if line i comes before line j
    in reading order."""
    order = np.zeros((len(lines),len(lines)),'B')
    def x_overlaps(u,v):
        return u[1].start<v[1].stop and u[1].stop>v[1].start
    def above(u,v):
        return u[0].start<v[0].start
    def left_of(u,v):
        return u[1].stop<v[1].start
    def separates(w,u,v):
        if w[0].stop<min(u[0].start,v[0].start): return 0
        if w[0].start>max(u[0].stop,v[0].stop): return 0
        if w[1].start<u[1].stop and w[1].stop>v[1].start: return 1
    for i,u in enumerate(lines):
        for j,v in enumerate(lines):
            if x_overlaps(u,v):
                if above(u,v):
                    order[i,j] = 1
            else:
                if [w for w in lines if separates(w,u,v)]==[]:
                    if left_of(u,v): order[i,j] = 1
            if j==highlight and order[i,j]:
                print((i, j), end=' ')
                y0,x0 = center(lines[i])
                y1,x1 = center(lines[j])
                plt.plot([x0,x1+200],[y0,y1])
    return order


def topsort(order):
    """Given a binary array defining a partial order (o[i,j]==True means i<j),
    compute a topological sort.  This is a quick and dirty implementation
    that works for up to a few thousand elements."""
    n = len(order)
    visited = np.zeros(n)
    L = []
    def visit(k):
        if visited[k]: return
        visited[k] = 1
        for l in find(order[:,k]):
            visit(l)
        L.append(k)
    for k in range(n):
        visit(k)
    return L


def array2pil(a):
    if a.dtype==np.dtype("B"):
        if a.ndim==2:
            return PIL.Image.frombytes("L",(a.shape[1],a.shape[0]),a.tostring())
        elif a.ndim==3:
            return PIL.Image.frombytes("RGB",(a.shape[1],a.shape[0]),a.tostring())
        else:
            raise TypeError("bad image rank")
    elif a.dtype==np.dtype('float32'):
        return PIL.Image.fromstring("F",(a.shape[1],a.shape[0]),a.tostring())
    else:
        raise TypeError("unknown image type")


def int2rgb(image):
    """Converts a rank 3 array with RGB values stored in the
    last axis into a rank 2 array containing 32 bit RGB values."""
    assert image.ndim==2
    assert isintarray(image)
    a = np.zeros(list(image.shape)+[3],'B')
    a[:,:,0] = (image>>16)
    a[:,:,1] = (image>>8)
    a[:,:,2] = image
    return a


def make_seg_white(image):
    assert isintegerarray(image),"%s: wrong type for segmentation"%image.dtype
    image = image.copy()
    image[image==0] = 0xffffff
    return image


def midrange(image,frac=0.5):
    """Computes the center of the range of image values
    (for quick thresholding)."""
    return frac*(np.amin(image)+np.amax(image))


def write_page_segmentation(fname,image):
    """Writes a page segmentation, that is an RGB image whose values
    encode the segmentation of a page."""
    assert image.ndim==2
    assert image.dtype in [np.dtype('int32'),np.dtype('int64')]
    a = int2rgb(make_seg_white(image))
    im = array2pil(a)
    im.save(fname)


def write_image_binary(fname,image,verbose=0):
    """Write a binary image to disk. This verifies first that the given image
    is, in fact, binary.  The image may be of any type, but must consist of only
    two values."""
    if verbose: print("# writing", fname)
    assert image.ndim==2
    image = np.array(255*(image>midrange(image)),'B')
    im = array2pil(image)
    im.save(fname)


def remove_noise(line,minsize=8):
    """Remove small pixels from an image."""
    if minsize==0: return line
    bin = (line>0.5*np.amax(line))
    labels,n = ndimage.label(bin)
    sums = ndimage.sum(bin,labels,range(n+1))
    sums = sums[labels]
    good = np.minimum(bin,1-(sums>0)*(sums<minsize))
    return good


def pad_image(image,d,cval=np.inf):
    result = np.ones(np.array(image.shape)+2*d)
    result[:,:] = np.amax(image) if cval==np.inf else cval
    result[d:-d,d:-d] = image
    return result


def extract(image,y0,x0,y1,x1,mode='nearest',cval=0):
    h,w = image.shape
    ch,cw = y1-y0,x1-x0
    y,x = np.clip(y0,0,max(h-ch,0)),np.clip(x0,0,max(w-cw, 0))
    sub = image[y:y+ch,x:x+cw]
    try:
        r = ndimage.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)
        if cw > w or ch > h:
            pady0, padx0 = max(-y0, 0), max(-x0, 0)
            r = ndimage.affine_transform(r, np.eye(2), offset=(pady0, padx0), cval=1, output_shape=(ch, cw))
        return r

    except RuntimeError:
        # workaround for platform differences between 32bit and 64bit
        # scipy.ndimage
        dtype = sub.dtype
        sub = np.array(sub,dtype='float64')
        sub = ndimage.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)
        sub = np.array(sub,dtype=dtype)
        return sub


def extract_masked(image,linedesc,pad=5,expand=0):
    """Extract a subimage from the image using the line descriptor.
    A line descriptor consists of bounds and a mask."""
    y0,x0,y1,x1 = [int(x) for x in [linedesc.bounds[0].start,linedesc.bounds[1].start, \
                  linedesc.bounds[0].stop,linedesc.bounds[1].stop]]
    if pad>0:
        mask = pad_image(linedesc.mask,pad,cval=0)
    else:
        mask = linedesc.mask
    line = extract(image,y0-pad,x0-pad,y1+pad,x1+pad)
    if expand>0:
        mask = ndimage.maximum_filter(mask,(expand,expand))
    line = np.where(mask,line,np.amax(line))
    return line


def glob_all(args):
    """Given a list of command line arguments, expand all of them with glob."""
    result = []
    for arg in args:
        if arg[0]=="@":
            with open(arg[1:],"r") as stream:
                expanded = stream.read().split("\n")
            expanded = [s for s in expanded if s!=""]
        else:
            expanded = sorted(glob.glob(arg))
        if len(expanded)<1:
            raise FileNotFoundError("%s: expansion did not yield any files"%arg)
        result += expanded
    return result


def remove_hlines(binary, scale, maxsize=10):
    labels, _ = ndimage.label(binary)
    objects = ndimage.find_objects(labels)
    for i, b in enumerate(objects):
        if width(b) > maxsize * scale:
            labels[b][labels[b] == i + 1] = 0
    _a = np.array(labels != 0, 'B')
    return np.array(labels != 0, 'B')


def binary_objects(binary):
    labels, n = ndimage.label(binary)
    objects = ndimage.find_objects(labels)
    return objects


def estimate_scale(binary):
    objects = binary_objects(binary)
    bysize = sorted(objects, key=area)
    scalemap = np.zeros(binary.shape)
    for o in bysize:
        if np.amax(scalemap[o]) > 0: continue
        scalemap[o] = area(o) ** 0.5
    scale = np.median(scalemap[(scalemap > 3) & (scalemap < 100)])
    return scale


def compute_boxmap(binary, scale, threshold=(.5, 4), dtype='i'):
    objects = binary_objects(binary)
    bysize = sorted(objects, key=area)
    boxmap = np.zeros(binary.shape, dtype)
    for o in bysize:
        if area(o) ** .5 < threshold[0] * scale: continue
        if area(o) ** .5 > threshold[1] * scale: continue
        boxmap[o] = 1
    return boxmap