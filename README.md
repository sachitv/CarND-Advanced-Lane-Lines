# Project Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/output_1_0.png "Before Calibration"
[image2]: ./examples/output_4_0.png "After Calibration"
[distorted]: ./examples/Distorted.png "Image with Distortion"
[undistorted]: ./examples/Undistorted.png "Image without Distortion"
[transformed]: ./examples/transformed.png "Transformed Image"
[laneDetected1]: ./examples/lane_detected_1.png "Detected Lane by method 1"
[laneDetected2]: ./examples/lane_detected_2.png "Detected Lane by method 1"
[exampleoutput]: ./examples/my_example_output.jpg "Example output"
[video1]: https://youtu.be/zeHWxIG0pXw "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first 3 cells of the IPython notebook located "Project4".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Some of the images mention they are bad pictures because OpenCV failed to find the right number of corners in them

![alt text][image1]

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

##### Before Distortion, the image looks as such:
![alt text][distorted]

##### After Distortion, it looks like this
![alt text][undistorted]

As you can see there are some subtle differences you notice in the image especially along the edges. For example you can notice that the tree on the right side looks different

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
If you look in the second from last block, you will notice the pipeline I have used for the bulk of this project in the function processImage()

The function getCombinedSobelMask() computes this information.

As you can see, I have used a combination of Sobel Masks in the X Direction on the S and V Channels as well as a Colour filter to generate the resultant mask that I use for my processing step. Note that although Udacity expects us to do this step before transformation, I found that in my case it was better to perform it after transformation. The reason I did so is that after projection it becomes much easier to highlight the lane lines we are looking for.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The function getROIImage() produces the transformed image that I am looking for here. You will notice that I also applied a gaussian filter to the image before this transformation. My intention in this case was to rule out noise in the image. This seemed to ensure I had less problems detecting features later in the pipeline.

In order to warp my image I used the following points

```
up = 0.61*height;
bottom_left = (0.10*width, height);
mid_left = (0.47*width,up);
mid_right = [0.525*width,up];
bottom_right = [0.94*width, height];

src = np.array([[bottom_left,mid_left,mid_right,bottom_right]], dtype=np.float32);

dst_height = math.floor(height)
dst_width = math.floor(width)
dst_dev_X = 250

dst_bottom_left = (dst_dev_X, dst_height);
dst_mid_left = (dst_dev_X,0);
dst_mid_right = [dst_width - dst_dev_X,0];
dst_bottom_right = [dst_width - dst_dev_X, dst_height];

dst = np.array([[dst_bottom_left,dst_mid_left,dst_mid_right,dst_bottom_right]], dtype=np.float32);


```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 128, 720      | 250, 720      | 
| 602, 439      | 250, 0        |
| 672, 439      | 1030, 0       |
| 1203, 720     | 1030, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][transformed]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There's two ways I did this.

1. With no prior knowledge of where the road is.
2. With prior knowledge of where the road is.

Let's go over each of them in sequence

**1. With no prior knowledge of where the road is.**

In order to do this, I performed the following steps
1. Take the lower half of the image and compute where we get the best response in terms of the number of white pixels.
2. We should have what appear to be 2 local maximas. These become the 'start' position for the next step
3. At the locations detected here, at the bottom of the image, let's draw a window of a certain width and height. In my case I used a width of 128 and height of 720. Basically 1/10 the dimensions in both directions of the input image
4. Let's also take an arbitrary deviation over which we can slide the window in this direction. In my case I chose 75 pixels.
5. Initialize the window to contain 1's in each pixel.
6. Slide the window starting from (start - deviation) to (start + deviation) for both the lanes 
7. While sliding across the image, compute an 'AND' operation of the window and the corresponding data in the input image.
8. Count the number of ones that we get in the result
9. If this number is greater than a threshold (4% of the area of the window) perform the following
    1. If this is greater than the largest value seen, save out the value and the corresponding position
    2. If this is equal to the largest value seen and the largest value was the previous value
        1. Count the number of times we have encountered such a scenario
        2. Take an average across the points in this range. This will ensure we get the mid point of places with equal sum. This is a small optimization that incredibly boosts the ability of the lane to fit better
        3. Set this value as our best value
10. Now, move this window up by the 'window height' number of pixels, keeping the x position locked to the last best value and perform the sliding search again.
11. When we are done with the whole image, save out all the points that we traversed.
12. Using these points as a start, I draw rectangles of the window size and highlight the pixels having values 1 as Red if they are on the left lane and blue if they are on the right lane
13. I also save the indices of these points.
14. I use the np.polyfit() to calculate the best fitting second degree polynomial for this data on both the left and right sides.
15. Draw the obtained polynomial across the height of the image. We now have our lanes detected.

My resultant Image looked as follows :

![alt text][laneDetected1]

**2. With prior knowledge of where the road is.**
Using the data from the last step, in my next frame of the video, I know where to look for the lanes. The steps are as follows:

1. Get non zero pixels within a certain distance of the last polynomial.
2. Categorize these appropriately for left and right lanes
3. Now, use the categorized data to fit a new polynomial to this data similar to what I did above.

This works pretty well, except in some rare occasions where the frames vary by a large amount. In order to catch these conditions I perform an XOR operation of the current and the previous frames. If the accumulated result is greater than a particular threshold, I resort to the first method to re calculate the curve.

The output should look something like this (taking another image here - you get the point)

![alt text][laneDetected2]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I wrote a function called getCurvature() in the State class that calculates this for each state and each lane.
It initially calculates the values in the perspective space and that is then translated to the world space using some approximation parameters xm_per_pix and ym_per_pix. These correspond to the number of meters in the real world in terms of pixels in the perspective space

It looks like this:

```python
def getCurvature(self, fit_curve, ploty, xvalues):
        #Evaluate it closest to the car at the maximum height in the image
        y_eval = np.max(ploty)
        curverad = ((1 + (2*fit_curve[0]*y_eval + fit_curve[1])**2)**1.5) / np.absolute(2*fit_curve[0])
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 50/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty*ym_per_pix, xvalues*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        
        return curverad
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

You can choose to run my code in single image output mode by switching the last line to 
```python
processSingleImage()
```
instead of 
```python
processVideo()
```
![alt text][exampleoutput]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
There are several issues I encountered while solving this project

1. Noise in the masked image.
2. Vanishing colours on lighter roads
3. Oversaturated images
4. Other cars on the road

Some possible things I can think of to improve my pipeline are:
1. I could try to do some cool stuff like boosting the contrast of the projected image before performing my search on it since it will better highlight the edges
2. In case there's too much of a difference between the left and right radii, I force calculate the lanes using method 1
3. Use a better combined mask than the one I used that helps filter out noise
