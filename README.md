# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture.png "Model Visualization"
[image2]: ./images/center.png "Center Camera Image"
[image3]: ./images/left.png "Left Camera Image"
[image4]: ./images/right.png "Right Image"
[image5]: ./images/flip.png "Flip Image"
[image6]: ./images/crop_center.png "Crop Image"
[image7]: ./images/resized_center.png "Resized Image"
[image8]: ./images/yuv_center.png "Crop Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 is the record of my autnomous driving
* images is the folder with all the images

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the convolutional network from nvidia and it worked out very well. I only included at the top of the network a Lambda layer for normalization and between my last convolutional layer and flatten layer a dropout layer with rate of 0.5. I also shaped the images to the given 66x200x3 shape. I used for activation elu because i wanted to introduce non linearity.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 18, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting plus I used ELU activation for every layer but dropout layer to introduce non-linearity.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also drove the car into the other direction for more variaty. I also used the second track to go against overfitting.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a good model was to use the Nivida architecture since it has been proven to be very successful in self-driving car tasks. I would say that this was the easiest part since a lot of other students have found it successful, the architecutre was recommended in the lessons and it's adapted for this use case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in ratio 4:1. Since I was using data augmentation techniques, the mean squared error was low both on the training and validation steps.

The hardest part was getting the data augmentation to work. I had a working solution early on but because there were a lot of syntax errors and minor omissons, it took a while to piece everything together. One of the problems, for example, was that I was incorrectly applying the data augmentation techniques and until I created a visualization of waht the augmented images looked like, I was not able to train a model that drives successfully around the track.


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also used the left and right camera with correction of -/+ 0.2 degree of sttering angle. Here are two examples of right and left camera:

![alt text][image3]
![alt text][image4]

To augment the data set, I also flipped images and angles thinking that this would balance the data set.For example, here is an image that has then been flipped:

![alt text][image5]

Then i preprocessed the dataset in my generator with the following technics
* cropped
* resized
* rgb2yuv
For example, here are the images:

![alt text][image6]
![alt text][image7]
![alt text][image8]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

### Video of the solution!

[![Video of my solution](https://img.youtube.com/vi/3pCsBfpuf0I/0.jpg)](https://youtu.be/3pCsBfpuf0I)
