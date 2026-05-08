import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk_t

from skimage.measure import profile_line
 
# Initialize Image class. Parameters:
# 1) image - greyscale image represented as numpy array
# 2) rotation - rotation to be applied to image in degrees
# 3) pxpermm_x - image scale in pixels per mm of x
# 4) pxpermm_y - Image scale in pixels per mm of y (if not specified scale x == scale y)
# 5) flipud - Whether to flip the image up-down
# 6) fliplr - Whether to flip the image left-right

##
## 5) mask = list format, e.g. [0,1]
## old param???
##

class Image:
    """
    Image class for managing image coordinates, scaling, and plotting.
    """

    def __init__(self, image, rotate, pxpermm_x, pxpermm_y = None, flipud = False, fliplr = False):
        """
        Initialize Image class.
        
        :param image: Greyscale image represented as numpy array
        :type image: np.ndarray
        
        :param rotate: Rotation to be applied to image in degrees
        :type rotate: float
        
        :param pxpermm_x: Image scale in pixels per mm of x
        :type pxpermm_x: float
        
        :param pxpermm_y: Image scale in pixels per mm of y
        :type pxpermm_y: float or None, default: None
        
        :param flipud: Whether to flip the image up-down
        :type flipud: bool, default: False
        
        :param fliplr: Whether to flip the image left-right
        :type fliplr: bool, default: False
        
        :return: No return
        :rtype: None
        """
        self.im = sk_t.rotate(image, rotate, resize=False)
        if flipud:
            self.im = np.flipud(self.im)
        if fliplr:
            self.im = np.fliplr(self.im)
        if pxpermm_y:
            self.sc_x = pxpermm_x
            self.sc_y = pxpermm_y
        else:
            self.sc_x = pxpermm_x
            self.sc_y = pxpermm_x
 
        self.o = np.array([0., 0.])
        self.shape = image.shape
        self.r = rotate

    def mask(self, threshold):
        """
        Masks the image below a given threshold.
        
        :param threshold: Minimum value to keep
        :type threshold: float
        
        :return: No return
        :rtype: None
        """
        im = self.im.copy()
        im[im < threshold] = 0

        self.im = im
 
        
    def px_to_mm(self, p_px):
        """
        Calculates position of point in mm, given position in px.
        
        :param p_px: Position in pixels [x, y]
        :type p_px: list or np.ndarray
        
        :return: Position in mm
        :rtype: np.ndarray
        """
        h = self.shape[0]
        p = np.array(p_px, dtype=np.float64)
        p *= np.array([1., -1.]) #Convert handedness 
        p += np.array([0., h]) #Translate origin to BL corner
        p[0] = p[0]/self.sc_x
        p[1] = p[1]/self.sc_y
        p -= self.o
        return p
    def mm_to_px(self, p_mm):
        """
        Calculates position of point in px, given position in mm.
        
        :param p_mm: Position in mm [x, y]
        :type p_mm: list or np.ndarray
        
        :return: Position in pixels
        :rtype: np.ndarray
        """
        h = self.shape[0]
        p = np.array(p_mm)
        p += self.o
        p[0] = p[0]*self.sc_x
        p[1] = p[1]*self.sc_y        
        p *= np.array([1., -1.]) #Convert handedness 
        p += np.array([0., h]) #Translate origin to TR corner
        return np.array(p, dtype=np.int64)
    def set_origin(self, p_px):
        """
        Sets origin of image from value in pixels.
        
        :param p_px: Position of the origin in pixels [x, y]
        :type p_px: list or np.ndarray
        
        :return: No return
        :rtype: None
        """
        self.o  = np.array([0., 0.])
        p_mm = self.px_to_mm(p_px)
        self.o = p_mm
        self.o_px = p_px
    def get_origin(self):
        """
        Returns the position of the origin in pixels.
        
        :return: Position of the origin in pixels
        :rtype: np.ndarray
        """
        o = np.array([0., 0.])
        o_px = self.mm_to_px(o)
        return o_px
    def plot_mm(self, ax, multiply_by = None, mask = None, extent = None, **kwargs):
        """
        Plot image with axes in physical units. kwargs are passed to plt.imshow method.
        
        :param ax: Matplotlib axis to plot on
        :type ax: matplotlib.axes.Axes
        
        :param multiply_by: Factor to multiply image values by
        :type multiply_by: float or None, default: None
        
        :param mask: Mask to apply
        :type mask: list or None, default: None
        
        :param extent: Custom extent
        :type extent: list or None, default: None
        
        :return: The plotted image object
        :rtype: matplotlib.image.AxesImage
        """
        x0 = 0
        x1 = self.im.shape[1]
        y0 = 0
        y1 = self.im.shape[0]
        x0, y0 = self.px_to_mm([x0, y0])
        x1, y1 = self.px_to_mm([x1, y1])
        if extent:
            self.extent = extent
        else:
            self.extent = [x0, x1, y1, y0]
        img = self.im
        if multiply_by:
            if mask:
                if len(mask) > 1:
                    self.masked_im = np.ma.masked_outside(img, mask[0], mask[1])
                    return ax.imshow(self.masked_im*multiply_by, extent = self.extent, **kwargs)
                else:
                    self.masked_im = np.ma.masked_less_equal(img, mask[0])
                    return ax.imshow(self.masked_im*multiply_by, extent = self.extent, **kwargs)
            else:
                return ax.imshow(img*multiply_by, extent = self.extent, **kwargs)
        else:
            return ax.imshow(img, extent = self.extent, **kwargs)
 
 
    def plot_px(self, ax, **kwargs):
        """
        Plot image with axes pixels. kwargs are passed to plt.imshow method.
        
        :param ax: Matplotlib axis to plot on
        :type ax: matplotlib.axes.Axes
        
        :return: The plotted image object
        :rtype: matplotlib.image.AxesImage
        """
        return ax.imshow(self.im, **kwargs)
 
    def plot_mm_split(self, ax, channel = 'b', multiply_by = None, mask = None, extent = None, **kwargs):
        ''' Plot image with axes in physical units and split channels. kwargs are passed
        to plt.imshow method.'''
 
        x0 = 0
        x1 = self.im.shape[1]
        y0 = 0
        y1 = self.im.shape[0]
        x0, y0 = self.px_to_mm([x0, y0])
        x1, y1 = self.px_to_mm([x1, y1])
        if extent:
            self.extent = extent
        else:
            self.extent = [x0, x1, y1, y0]
        b, g, r    =    self.im[:, :, 0], self.im[:, :, 1], self.im[:, :, 2]
 
        if channel == 'b':
            img = b
        elif channel == 'g':
            img = g
        elif channel == 'r':
            img = r
        if multiply_by:
            if mask:
                self.masked_im = np.ma.masked_less_equal(img, mask)
                return ax.imshow(self.masked_im*multiply_by, extent = self.extent, **kwargs)
            else:
                return ax.imshow(img*multiply_by, extent = self.extent, **kwargs)
        else:
            return ax.imshow(img, extent = self.extent, **kwargs)
 
        
    def profile_mm(self, src_mm, dst_mm, width_mm, **kwargs):
        """
        Extracts a line profile from the image.
        
        :param src_mm: Source point in mm [x, y]
        :type src_mm: list or np.ndarray
        
        :param dst_mm: Destination point in mm [x, y]
        :type dst_mm: list or np.ndarray
        
        :param width_mm: Width of the profile to extract
        :type width_mm: float
        
        :return: Tuple of physical coordinates and extracted profile
        :rtype: tuple
        """
        src_px = np.flip( self.mm_to_px(src_mm) )
        dst_px = np.flip( self.mm_to_px(dst_mm) )
        width_px = int(width_mm*self.sc_x)
        p = profile_line(self.im, src_px, dst_px, linewidth=width_px, **kwargs)
        r = np.linspace(src_mm, dst_mm, len(p))
        return r, p
    def create_im(self, im):
        """
        Creates a new Image object with the same origin and scale.
        
        :param im: New image data
        :type im: np.ndarray
        
        :return: New Image object
        :rtype: Image
        """
        out = Image(im, 0., self.sc_x)
        out.set_origin(self.o_px)
        return out