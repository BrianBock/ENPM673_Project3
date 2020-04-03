# ENPM673_Project3

Justin Albrecht and Brian Bock

## Project Description
This project uses a Guassian Mixture Model to probabilisitically learn and then detect colors. The sample video, `detectbouy.avi` (viewable in the media folder) shows the view from an underwater camera in a large pool with 3 bouys of different colors - yellow, orange, and green.

![3 bouys](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/3_bouys.gif)

## Packages Required

This entire project is written in Python 3.7 and requires the following packages:

`numpy`, `cv2`, `matplotlib`, `random`, `os`, `mpl_toolkits`, `math`, `scipy`
## How to Run
Clone the entire directory. All files are required. 


## Data Selection - Region of Interest (ROI)

We manually selected the region of interest for each bouy for each frame of the video. 
[![Bouy Selection](https://github.com/BrianBock/ENPM673_Project3/blob/master/images/bouy_select.gif)](https://youtu.be/gAHzZghxUaw)

Click on the gif above to see an extended video of the bouy ROI selection process.


## Gaussian Mixture Modeling

