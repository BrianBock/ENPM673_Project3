# ENPM673_Project3

Justin Albrecht and Brian Bock

## Project Description
This project uses a Gaussian Mixture Model to probabilisitically learn and then detect colors. The sample video, `detectbouy.avi` (viewable in the media folder) shows the view from an underwater camera in a large pool with 3 bouys of different colors - yellow, orange, and green.

![3 bouys](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/3_bouys.gif)

## Packages Required

This entire project is written in Python 3.7 and requires the following packages:

`numpy`, `cv2`, `matplotlib`, `random`, `os`, `mpl_toolkits`, `math`, `scipy`

## How to Run
Clone the entire directory. All files are required. 


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

We're only ever working with one color buoy at a time, and all the steps are then repeated for the subsequent buoy colors. It is therefore implicit when we say "buoy images" that we are referring to the buoy images of a particular color (yellow, orange, green).



We take all of the buoy images and flatten them into three 1 dimensional arrays, one for each color channel, which are then stacked into a 3-dimensional array for each image. By selecting non-zero elements, we eliminate the black border from our elliptical mask. The flattened stacked buoy images are combined, so we have one huge *Nx3* numpy array, where there are *N* total pixels in all of the buoy images, and *3* is the number of color channels. This *N* is used many times in the next section, and always refers to this number (for each color buoy).



## Gaussian Mixture Modeling

