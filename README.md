# Low-Level-Image-Processing-Gaussian-Blurr
•	The program is written in Python 3.7.
•	The purpose of this program is to apply low-level filtering techniques to an image to get a smooth curve for further processing.
•	Gaussian filter was created of n*n (decided by the user), which was later convolved through the image to apply the filter.

# Input/Output:
•	The program loads an image file provided on the command line and decompress it into a numpy array.
•	Split the input image into 3 channels (R, G, B)
•	Compute a two-dimensional isotropic Gaussian kernel.
•	Convolve the Gaussian kernel with each channel of the image.
•	Save the result.

# Notes:
•	With a smaller kernel for example 3*3, it looks like the image does not get blurred as expected as the value for sigma is higher. I had to use a kernel of size 9*9 with a 4.0 value for sigma to get desired result. Any suggestions about it? 

# Running the program
•	Install Libraries:
pip3 install numpy
pip3 install imageio
pip3 install PIL

•	python3 hw1.py --k value --sigma value input_image.jpg output_image.jpg
