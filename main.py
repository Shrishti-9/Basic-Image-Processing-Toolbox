import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st 

# Function to Convert PIL Image to OpenCV Format
def pil_to_cv2(img):
    return np.array(img.convert("RGB"))[:, :, ::-1]

# Function to Convert OpenCV Image to PIL Format
def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Functions to apply Image Processing Techniques
def process_image(image, technique):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    
    if technique == "Grayscale":
        return cv2_to_pil(gray_img)
    
    elif technique == 'Gaussian Blur':
        gaussian_img = cv2.GaussianBlur(gray_img, (5,5), 0)
        return cv2_to_pil(gaussian_img)
        
    elif technique == 'Median Blur':
        median_img = cv2.medianBlur(gray_img, 5)
        return cv2_to_pil(median_img)
        
    elif technique == 'Bilateral Filter': #Preserve Edges
        bilateral_img = cv2.bilateralFilter(gray_img, 9, 75, 75)
        return cv2_to_pil(bilateral_img)
    
    elif technique == 'Canny Edge Detection':
        canny_img = cv2.Canny(gray_img, 100, 200)
        return cv2_to_pil(canny_img)
    
    elif technique == 'Sobel X Edge Detection':
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
        sobelx = np.uint8(np.abs(sobelx))
        return cv2_to_pil(sobelx)
    
    elif technique == 'Sobel Y Edge Detection':
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
        sobely = np.uint8(np.abs(sobely))
        return cv2_to_pil(sobely)
    
    elif technique == 'Laplacian Edge Detection':
        laplacian_img = cv2.Laplacian(gray_img, cv2.CV_64F)
        laplacian_img = np.uint8(np.abs(laplacian_img))
        return cv2_to_pil(laplacian_img)
        
    elif operation == "Binary Thresholding":
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        return cv2_to_pil(binary)
    
    elif operation == "Adaptive Thresholding":
        adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2_to_pil(adaptive)
    
    elif operation == "Otsu's Thresholding":
        _, otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2_to_pil(otsu)
    
    elif operation == "Histogram Equalization":
        equalized = cv2.equalizeHist(gray_img)
        return cv2_to_pil(equalized)
    
    elif operation == "Contour Detection":
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        return cv2_to_pil(contour_img)
    
    elif operation == 'Dilation':
        dilated = cv2.dilate(gray_img, kernel, iterations=1)
        return cv2_to_pil(dilated)

    elif operation == 'Erosion':
        eroded = cv2.erode(gray_img, kernel, iterations=1)
        return cv2_to_pil(eroded)

    elif operation == 'Opening':    # Opening (Erosion followed by Dilation)
        opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
        return cv2_to_pil(opening)

    elif operation == 'Closing':  # Closing (Dilation followed by Erosion)
        closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
        return cv2_to_pil(closing)
              
    return cv2_to_pil(image)

st.title('Image Processing Toolbox')
st.write('Apply various image processing techniques')

uploaded_file = st.file_uploader('Upload ann Image', type=['jpg', 'jpeg', 'png', 'jfif'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cv2_image = pil_to_cv2(image)
    
    # Select Processing Option
    operation = st.selectbox("Choose an Operation", [
        "Original", "Grayscale", "Gaussian Blur", 'Median Blur', 'Bilateral Filter', "Canny Edge Detection",
        "Sobel X Edge Detection", "Sobel Y Edge Detection", 'Laplacian Edge Detection', "Binary Thresholding", "Adaptive Thresholding",
        "Otsu's Thresholding", "Histogram Equalization", "Contour Detection", 'Dilation', 'Erosion', 'Opening', 'Closing'
    ])
    
    # Process Image
    if operation == "Original":
        output_img = image
    else:
        output_img = process_image(cv2_image, operation)
    
    # Display Images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(output_img, caption=f"{operation} Applied", use_column_width=True)

    # Download Processed Image
    st.download_button(
        label="Download Processed Image",
        data=output_img.tobytes(),
        file_name="processed_image.png",
        mime="image/png"
    )


# Save the file as main.py and run:
# streamlit run main.py
