## Vehicle Detection Project
### In this project, main goal is to write a software pipeline to detect vehicles in a video.

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Histogram of Oriented Gradients (HOG)

#### 1. Extracted HOG features from the training images.

The code for this step is contained in the code cell 2  of the IPython notebook 

I started by reading in all the `vehicle` and `non-vehicle` images in lists. And then shuffling the lists to generalize the data.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


I then explored different color spaces like HSV,HLS,YCrCb, YUV and finally settled with YUV and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes as I applied shuffle and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` and hog channel `0`:


![alt text][image2]



#### 2. Final choice of HOG parameters and color features

I tried various combinations of parameters like colorspace (HSV,HLS,YCrCb, YUV) ; orientations (8,10,11,12,16) ; pixel per cell(8 to 24); cell per block (2) ; hog channel (0,1,'ALL') ; spatial size ((16, 16) and (32,32)) ; histbins (16 to 32)

At first I was getting many false positives especiall in shadow areas. I tried to reduce it using  YUV and YCrCb colorspace. Also, hog channel ('ALL') was not providing good results. Then I settled using hog channel 0. Also , I found out if we use YUV or LUV colorspace, it returns np.sqrt error when using hog channel 1 or 2 or 'ALL' . Then I found out it was due to negative pixel values . It was resolved when I put  transform_sqrt=False of skimage.hog() when using these colorspaces. 

Finally I settled with below parameters:
color_space = 'YUV'
orient = 12  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16


#### 3.Training a classifier using selected HOG features.

I trained a linear SVM for cars and non car images in code cell 15. Using hog, spatial and histogram color features I trained the classifier for car and non car image with an 80-20 split of data as training and test data, to return an accuracy of 99.46%. A StandardScaler from sklearn.preprocessing to normalise the feature-set and avoid favouritism in features. Also I tried tuning the C(penalty) parameter of SVM classifier and finally settled with C=100.0

### Sliding Window Search

#### 1.Implementing a sliding window search.

I implemented a sliding window search in the lower half of the image from 400 to 650, as cars would be found only in the lower half of the image, not the sky. I implemented the code for it using the helper functions provided in the classroom material. Additionally, I incremented the width and height by 16 px, after each row-wise pass.But that too didnt gave good results. So I applied a windows size of 64px.

![alt text][image3]

#### 2. Examples of test images to demonstrate pipeline 

Ultimately I searched on two scales using YCrCb 1-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1.Link to the final video output. 
Here's a [link to my video result](./project_video.mp4)


#### 2. Implementation of filter for false positives and method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. An average of the past 10-15 frames was used with a threshold of 10. The heatmap technique helped eliminate false positives and allowed for a more consistent detection over the frames, removing wobblines in the bounding box.

Here's an example result showing the heatmap from a series of frames of video, the result of scipy.ndimage.measurements.label() and the bounding boxes then overlaid on the last frame of video:

### Here are 3 frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 3 frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Problems/issues:
- It took a  lot of time for parameter tuning. Still was not able to fully eliminate false positives.
- Processing time for the video was more. Tried implementing hog subsampling but it lead to even more increase in processing time.
- An elevation in road can change the output.
- Lane change can lead to changes in output. Need to make more robust pipeline.

Improvements:
A CNN approach or more specifically YOLO architecture can help to make more accurate detections with less data preprocessing and parameter tuning.
I am currently working on that.

