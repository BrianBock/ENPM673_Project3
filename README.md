# ENPM673 Project 3

Justin Albrecht and Brian Bock

## Project Description
This project uses a Gaussian Mixture Model to probabilisitically learn and then detect colors. The sample video, `detectbouy.avi` (viewable in the media folder) shows the view from an underwater camera in a large pool with 3 bouys of different colors - yellow, orange, and green.

![3 bouys](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/3_bouys.gif)

## Packages Required

This entire project is written in Python 3.7 and requires the following packages:
	
	numpy
	cv2
	matplotlib
	scipy.stats
	math
	matplotlib
		matplotlib.patches
	mpl_toolkits.mplot3d
	random
	os


## Instructions for Running the Program
Clone the entire repository (all files are required) to a directory you have write access to. We've included our trained Thetas as part of our submission, to save you the time intensive process of re-running that. The program is therefore pre-trained and ready to run. Open a new terminal window and navigate to the repository directory. Type `python buoy_detection.py`. If you have additional versions of python installed you may need to run `python3 buoy_detection.py` instead. 




# How it Works

## Data Processing

### Data Selection - Region of Interest (ROI)

We manually selected the region of interest for each bouy for each frame of the video. 
[![Bouy Selection](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/bouy_select.gif)](https://youtu.be/gAHzZghxUaw)

Click on the gif above to see an extended video of the bouy ROI selection process.
The frame is copied and cropped to the ROI, and then masked in an elliptical shape with major and minor axes equal to the width and height of the rectangular ROI. The elliptical masking eliminates the water captured in the rectangular ROI around the spherical buoy. This rounded image is then saved into it's respective color folder. This process was repeated for each buoy in every frame. The yellow buoy is visible throughout the entire video, while the orange buoy is off-screen for a few frames, and the green buoy is only visible for the first 43 frames. 

![Yellow buoy](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/yellow173.png)
![Orange buoy](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/orange175.png)
![Green buoy](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/green37.png)

### Division into Training and Testing Data

The rounded buoy images are then randomly divided into Training and Testing data folders such that the ratio of Training to Testing images (for each color) is approximately 70:30. The breakdowns for each color are listed below.

**Color** | **Num of Training Images** | **Num of Testing Images**
:---: | :---: | :---:
Yellow | 140 | 60
Orange | 124 | 54
Green | 31 | 14


### Reading in image data for GMM

**Notes:**
Recognizing that our data might be more distinct in other color spaces, we built-in the option to work in either BGR or HSV color space, which can be changed via a simple toggle. For the sake of brevity, we'll be referring to image color channels as just BGR (regardless of the mode) so that we don't need to refer to both BGR and HSV. The steps are equally applicable for each color space. 





## Gaussian Mixture Modeling





## Buoy Detection

### Color Segmentation
We use our pre-trained data and run our Test images through it. 
Our `determineThresholds` function runs our Test images through our Trained Gaussians, and produces the probability that each pixel belongs to those Gaussians. The probabilities are summed over each of our $K$ Gaussians. Using an experimentally determined threshold, we take a certain percentage of those probabilities and define them as the our certainty for each buoy color. 

We return to looking at the entire original video, one frame at a time. We convert the frame into a flat 2 channel array, to make it easier to work with. We then create a new black image with the same dimensions as the original frame, which will be the canvas on which we draw new pixel. If any pixel in the original frame meets our buoy color thresh criteria, that pixel coordinate in the black image is colored to match the buoy's color. The first few frames of the video are very noisy with the pool sidewall closely matching several buoy colors. 


![Colored pixels based on Gaussians](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/all_colors.png)
### Contour Detection
Take our new image and create a binary image where any previously colored (non black) pixel becomes white:

![Binary Image](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/bin.png)


To reduce noise, we apply a small Gaussian blur on this image and then run `cv2.findContours`, which returns us all of the contours in the image:

![Rough Contours](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/rough_contours.png)

Many of these contours are tiny and erroneous, so we sort our contours and only keep the largest 8:

![8 Contours](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/8_contours.png)
 
We use `cv2.minEnclosingCircle` to find the center and radius of the minimum enclosing circle for each contour. We know that our buoys are mostly circular, and so the area enclosed by the contour around a buoy should be comparable to the area defined by that minimum enclosing circle. Any contour whose area differs from it's min enclosing circle area by less than an experimentally defined threshold is likely a buoy. We then identify the color contained within that circle, and use it to define the color of the circular contour we draw:

![Buoys Traced with Colored Circles](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/ring_contours.png)

You have the option to output the data with the rough drawn Gaussian determined pixel colors. For cleaner output, you can toggle `solid=True` in the top of `buoy\_detection.py`, which will render solid circles on a black canvas:

![Buoys Replaced with Solid Circles](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/contours.png)


You can view both of these output versions in the Videos section. The new frame is exported side by side with the original video for side by side viewing:

![Final Frame](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/finalframe.png)



# Final Output
![Color Rings](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/color_rings.gif)
![Solid Circles](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/solid_buoys.gif)

Buoys with Colored Rings:https://youtu.be/UGUHVrVdI2U

Bupys with Colored Circles: https://youtu.be/aP2YAfDPRnQ