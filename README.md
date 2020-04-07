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


# Final Output
![Color Rings](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/color_rings.gif)
![Solid Circles](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/solid_bouys.gif)

Buoys with Colored Rings:https://youtu.be/UGUHVrVdI2U

Bupys with Colored Circles: https://youtu.be/aP2YAfDPRnQ