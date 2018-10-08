# **Behavioral Cloning** 

## Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/layers.png "layers"
[image2]: ./examples/center.jpg "center"
[image3]: ./examples/left.jpg "left"
[image4]: ./examples/right.jpg "right"
[image5]: ./loss1.png "loss1"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model1.h5 containing a trained convolution neural network 
* run1-1.mp4 Video of driving the car in autonomous mode for track one
* run1-2.mp4 Video of driving the car in autonomous mode for track two
* report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model1.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the model from Nvidia's [paper](https://arxiv.org/abs/1604.07316). 

#### 2. Attempts to reduce overfitting in the model
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 63 - 70). The ratio between training and validation data is 4 : 1. Moreover, I used dropout layer to reduce overfitting. In the end, The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also collected data for both clock-wise and counter-clockwise and I used data from both track one and track two. Moreover, I used several data augmention tricks like flipping, cropping and images from different cameras.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduce validation loss.

In the beginning, I implemented LeNet without data augmention and trained the model util validation loss (~3 epoches) got stable. But the simulation result is terrible. The car cannot make it in the first turn. After using all the data augmention tricks, the model can already complete the track one.

Then I switched to a deeper architecture as suggested (the Nvidia one). I found out I need more epoches (~8 epoches) to train. Both training and validation loss decreases, and the vehicle is still able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

It has a normalization layer, a cropping layer, six convolutional layers, a dropout layer, and four dense layers:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving in both clockwise and counter-clockwise. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to adjust the angles. Here are the examples:

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and add images from left and right cameras (code line 27 - 37, line 14). 

After the collection process, I had 5870 * 2 * 3 = 35,220 number of images. I then preprocessed this data by normalizing to range [-0.5,0.5] and cropping the image (code line 73 - 74).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by the loss curve below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image5]
