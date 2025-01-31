# Basic-Image-Processing-Toolbox

**1) Image Blurring (Image Smoothing):-** Image blurring is achieved by convolving the image with a low-pass filter kernel. It is useful for removing noise. It actually removes high frequency content (eg: noise, edges) from the image. 
**a) Gaussian Blurring :-** In this method, a Gaussian kernel is used. We should specify the width and height of the kernel which should be positive and odd. We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively. If only sigmaX is specified, sigmaY is taken as the same as sigmaX. If both are given as zeros, they are calculated from the kernel size. Gaussian blurring is highly effective in removing Gaussian noise from an image. a Gaussian filter takes the neighbourhood around the pixel and finds its Gaussian weighted average. This Gaussian filter is a function of space alone, that is, nearby pixels are considered while filtering. It doesn't consider whether pixels have almost the same intensity. It doesn't consider whether a pixel is an edge pixel or not. So it blurs the edges also, which we don't want to do.
Function Syntax - **dst =   cv2.GaussianBlur(   src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]   )**
Parameters-
src input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
dst output image of the same size and type as src.
ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma.
sigmaX Gaussian kernel standard deviation in X direction.
sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively; to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
borderType pixel extrapolation method.

**b) Median Blurring :-** The function takes the median of all the pixels under the kernel area and the central element is replaced with this median value. This is highly effective against salt-and-pepper noise in an image. Interestingly, in the above filter, the central element is a newly calculated value which may be a pixel value in the image or a new value. But in median blurring, the central element is always replaced by some pixel value in the image. It reduces the noise effectively. Its kernel size should be a positive odd integer.
Function Syntax - **dst =   cv2.medianBlur( src, ksize[, dst]   )**
Parameters-
src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
dst destination array of the same size and type as src.
ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5,7, ...

**c) Bilateral Filtering :-** It is highly effective in noise removal while keeping edges sharp. But the operation is slower compared to other filters.Bilateral filtering also takes a Gaussian filter in space, but one more Gaussian filter which is a function of pixel difference. The Gaussian function of space makes sure that only nearby pixels are considered for blurring, while the Gaussian function of intensity difference makes sure that only those pixels with similar intensities to the central pixel are considered for blurring. So it preserves the edges since pixels at edges will have large intensity variation.
Function Syntax - **dst =   cv2.bilateralFilter(    src, d, sigmaColor, sigmaSpace[, dst[, borderType]] )**
Parameters-
src Source 8-bit or floating-point, 1-channel or 3-channel image.
dst Destination image of the same size and type as src .
d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
sigmaColor Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
borderType border mode used to extrapolate pixels outside of the image.

**2) Edge Detection :-** It is a technique used to identify the boundaries of objects within images. It helps in simplifying the image data by reducing the amount of information to be processed while preserving the structural properties of the image. This simplification is essential for various image analysis tasks, including object recognition, segmentation, and image enhancement.

**a) Canny Edge Detection :-** Canny edge detection is the most widely-used edge detector. For many of the applications that require edge detection, Canny edge detection is sufficient.
Canny edge detection has the following three steps:
**Gradient calculations:** Edges are pixels where intensity changes abruptly. From previous modules, we know that the magnitude of gradient is very high at edge pixels. Therefore, gradient calculation is the first step in Canny edge detection.
**Non-maxima suppression:** In the real world, the edges in an image are not sharp. The magnitude of gradient is very high not only at the real edge location, but also in a small neighborhood around it. Ideally, we want an edge to be represented by a single, pixel-thin contour. Simply thresholding the gradient leads to a fat contour that is several pixels thick. Fortunately, this problem can be eliminated by selecting the pixel with maximum gradient magnitude in a small neighborhood (say 3x3 neighborhood) of every pixel in the gradient image. The name non-maxima suppression comes from the fact that we eliminate (i.e. set to zero) all gradients except the maximum one in small 3x3 neighborhoods over the entire image.
**Hysteresis thresholding:** After non-maxima suppression, we could threshold the gradient image to obtain a new binary image which is black in all places except for pixels where the gradient is very high. This kind of thresholding would naively exclude a lot of edges because, in real world images, edges tend to fade in and out along their length. For example, an edge may be strong in the middle but fade out at the two ends. To fix this problem, Canny edge detection uses two thresholds. First, a higher threshold is used to select pixels with very high gradients. We say these pixels have a strong edge. Second, a lower threshold is used to obtain new pixels that are potential edge pixels. We say these pixels have a weak edge. A weak edge pixel can be re-classified as a strong edge if one of its neighbor is a strong edge. The weak edges that are not reclassified as strong are dropped from the final edge map.
Function Syntax - **edges   =   cv.Canny(   dx, dy, threshold1, threshold2[, edges[, L2gradient]]   )**
Parameters
dx 16-bit x derivative of input image (CV_16SC1 or CV_16SC3).
dy 16-bit y derivative of input image (same type as dx).
edges output edge map; single channels 8-bit image, which has the same size as image .
threshold1 first threshold for the hysteresis procedure.
threshold2 second threshold for the hysteresis procedure.
L2gradient a flag, indicating whether a more accurate L2 norm =âˆš(dI/dx)2+(dI/dy)2 should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ). If you want better accuracy at the expense of speed, you can set the L2gradient flag to true.

**b) Sobel Edge Detection :-** Sobel function for calculating the X and Y Gradients.
Function Syntax - **dst =   cv2.Sobel(  src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]] )**
Parameters-
src input image.
dst output image of the same size and the same number of channels as src .
ddepth output image depth,in the case of 8-bit input images it will result in truncated derivatives.
dx order of the derivative x.
dy order of the derivative y.
ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
scale optional scale factor for the computed derivative values; by default, no scaling is applied.
delta optional delta value that is added to the results prior to storing them in dst.
borderType pixel extrapolation method.

The X and Y Gradients are calculated using the Sobel function. Note that the depth of the output images is set to CV_32F because gradients can take negative values.

# Apply sobel filter along x direction
sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
# Apply sobel filter along y direction
sobely = cv2.Sobel(image,cv2.CV_32F,0,1)

**c) Laplacian Edge Detection :-** Laplacian Edge Detection is a technique in image processing used to highlight areas of rapid intensity change, which are often associated with edges in an image. Unlike gradient-based methods such as Sobel and Canny, which use directional gradients, Laplacian Edge Detection relies on the second derivative of the image intensity. Following are the steps for Edge Detection Using Laplacian
-> Convert the Image to Grayscale: Edge detection usually starts with a grayscale image to simplify computations.
-> Apply Gaussian Blur (Optional): Smoothing the image with a Gaussian blur can reduce noise and prevent false edge detection.
-> Apply the Laplacian Operator: Convolve the image with a Laplacian kernel to calculate the second derivative.
**laplacian = cv2.Laplacian(image, cv2.CV_64F)**

**3) Thresholding :-**
**a) Binary Thresholding :-** If the pixel value is smaller than or equal to the threshold, it is set to 0, otherwise it is set to a maximum value. The function cv.threshold is used to apply the thresholding. 
**retval, dst = cv.threshold(src, thresh, maxval, type[, dst])**
All simple thresholding types are:
cv.THRESH_BINARY
cv.THRESH_BINARY_INV
cv.THRESH_TRUNC
cv.THRESH_TOZERO
cv.THRESH_TOZERO_INV

**b) Adaptive Thresholding :-** In the previous section, we used one global value as a threshold. But this might not be good in all cases, e.g. if an image has different lighting conditions in different areas. In that case, adaptive thresholding can help. Here, the algorithm determines the threshold for a pixel based on a small region around it. So we get different thresholds for different regions of the same image which gives better results for images with varying illumination.
 Syntax: **cv2.adaptiveThreshold(source, maxVal, adaptiveMethod, thresholdType, blocksize, constant)**
Parameters-
-> source: Input Image array(Single-channel, 8-bit or floating-point)
-> maxVal: Maximum value that can be assigned to a pixel.
-> adaptiveMethod: Adaptive method decides how threshold value is calculated.
 cv2.ADAPTIVE_THRESH_MEAN_C: Threshold Value = (Mean of the neighbourhood area values – constant value). In other words, it is the mean of the blockSize×blockSize neighborhood of a point minus constant.
cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Threshold Value = (Gaussian-weighted sum of the neighbourhood values – constant value). In other words, it is a weighted sum of the blockSize×blockSize neighborhood of a point minus constant.
-> thresholdType: The type of thresholding to be applied.
-> blockSize: Size of a pixel neighborhood that is used to calculate a threshold value.
-> constant: A constant value that is subtracted from the mean or weighted sum of the neighbourhood pixels.

**c) Otsu's Thresholding :-** In global thresholding, we used an arbitrary chosen value as a threshold. In contrast, Otsu's method avoids having to choose a value and determines it automatically. Consider an image with only two distinct image values (bimodal image), where the histogram would only consist of two peaks. A good threshold would be in the middle of those two values. Similarly, Otsu's method determines an optimal global threshold value from the image histogram.
Syntax: **cv2.threshold(source, thresholdValue, maxVal, thresholdingTechnique)**
Parameters: 
-> source: Input Image array (must be in Grayscale).
-> thresholdValue: Value of Threshold below and above which pixel values will change accordingly.
-> maxVal: Maximum value that can be assigned to a pixel.
-> thresholdingTechnique: The type of thresholding to be applied.

**4) Histogram Equalization :-** Histogram Equalization is a non-linear method for enhancing contrast in an image. equalizeHist() performs histogram equalization on a grayscale image. The syntax is given below.
Function Syntax **dst =   cv2.equalizeHist(   src[, dst]  )**
Parameters-
src - Source 8-bit single channel image.
dst - Destination image of the same size and type as src.

**5) Contour Detection :-** Using contour detection, we can detect the borders of objects, and localize them easily in an image. It is often the first step for many interesting applications, such as image-foreground extraction, simple-image segmentation, detection and recognition. OpenCV makes it really easy to find and draw contours in images. It provides two simple functions: findContours() --> drawContours()
findContours() syntax :- **contours, hierarchy = cv2.findContours(image, mode, method)**
Parameters-
image: The binary input image obtained in the previous step.
mode: This is the contour-retrieval mode. We provided this as RETR_TREE, which means the algorithm will retrieve all possible contours from the binary image. More contour retrieval modes are available, we will be discussing them too. You can learn more details on these options here. 
method: This defines the contour-approximation method. In this example, we will use CHAIN_APPROX_NONE.Though slightly slower than CHAIN_APPROX_SIMPLE, we will use this method here tol store ALL contour points. 

drawContours() syntax :- **cv2.drawContours(image, contours, contourIdx, color, thickness, lineType)**
Parameters-
image: This is the input RGB image on which you want to draw the contour.
contours: Indicates the contours obtained from the findContours() function.
contourIdx: The pixel coordinates of the contour points are listed in the obtained contours. Using this argument, you can specify the index position from this list, indicating exactly which contour point you want to draw. Providing a negative value will draw all the contour points.
color: This indicates the color of the contour points you want to draw. We are drawing the points in green.
thickness: This is the thickness of contour points.

**Morphological Transformations :-** Morphological transformations are some simple operations based on the image shape. It is normally performed on binary images. It needs two inputs, one is our original image, second one is called structuring element or kernel which decides the nature of operation. Two basic morphological operators are Erosion and Dilation.
**a) Erosion :-** The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of foreground object (Always try to keep foreground in white). The kernel slides through the image (as in 2D convolution). A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero). All the pixels near boundary will be discarded depending upon the size of kernel. So the thickness or size of the foreground object decreases or simply white region decreases in the image. It is useful for removing small white noises, detach two connected objects etc.
**cv2.erode(image, kernel, iterations)**

**b) Dilation :-** It is just opposite of erosion. Here, a pixel element is '1' if at least one pixel under the kernel is '1'. So it increases the white region in the image or size of foreground object increases. Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won't come back, but our object area increases. It is also useful in joining broken parts of an object.
**cv2.dilate(image, kernel, iterations)**

**c) Opening :-** Opening is just another name of erosion followed by dilation. It is useful in removing noise.
**result = cv2.morphologyEx( src, cv2.MORPH_OPEN, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]] )**
Parameters-
src - Source image. The number of channels can be arbitrary. The depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
dst - Destination image of the same size and type as source image.
op - Type of a morphological operation
kernel - Structuring element. It can be created using getStructuringElement.
anchor - Anchor position with the kernel. Negative values mean that the anchor is at the kernel center.
iterations - Number of times erosion and dilation are applied.
borderType - Pixel extrapolation method.
borderValue - Border value in case of a constant border. The default value has a special meaning.

**d) Closong :-** Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
**result = cv2.morphologyEx( src, cv2.MORPH_CLOSE, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]] )**
