#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/figure_1.png "Model Visualization"
[image2]: ./writeup_images/figure_1-2.png "Grayscaling"
[image3]: ./writeup_images/figure_1-3.png "Recovery Image"
[image4]: ./writeup_images/figure_2.png "Recovery Image"
[image5]: ./writeup_images/figure_3.png "Recovery Image"
[image6]: ./writeup_images/s1.png "Normal Image"
[image7]: ./writeup_images/s2.png "Normal Image"
[image8]: ./writeup_images/s3.png "Normal Image"
[video1]: ./track.mp4 "Video"
  
---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5_3 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5_3
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of 5 convolution neural networks
The first one with 5x5 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Convolution Layer1		|**Filter**: 24, **kernel**: 5 * 5, **Stride**: 2 * 2, **Activation**: RELU
|Dropout     | 0.9
|  Convolution Layer2  	| **Filter**: 36, **kernel**: 5 * 5, **Stride**: 2 * 2, **Activation**: RELU	|
|Dropout     | 0.8				|	
|  Convolution Layer3  	| **Filter**: 48, **kernel**: 5 * 5, **Stride**: 2 * 2, **Activation**: RELU	|											|
|  Convolution Layer4  	| **Filter**: 64, **kernel**: 3 * 3, **Stride**: 2 * 2, **Activation**: RELU	|
|Dropout     | 0.8
|  Convolution Layer5  	| **Filter**: 64, **kernel**: 3 * 3, **Stride**: 2 * 2, **Activation**: RELU	|
| Flattern Layer|  				|
| Fully Connected Layer    |  Neurons: 100     									|
| Fully Connected Layer    |  Neurons: 50
| Fully Connected Layer    |  Neurons: 10
| Output    | 


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and add more training data on the area that vehicle behave bad, such as big curve and bridge area.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia model. I thought this model might be appropriate because they are using this model to do the similiar work and get pretty good result and the architecture is not complicated.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. The result is shown as below:

![alt text][image1]

To combat the overfitting, I modified the model with dropout layers so that the mean squared error of validation is getting lower. The final result is shown as below:

![alt text][image2]

Then I detected that vehicle always behave badly during big curve, so I add more training data for that scenario also using multiple cameras to let vehicle learning how to recovery from offset.

The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 105-121) consisted of a convolution neural network layers, cropout layer, dropout layer and fully connected layer.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to recovery from offset. These images show what a recovery looks like starting from :

![alt text][image7]
![alt text][image8]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

To release useless imformation, I applied an crop layer to cutting sky and lower side of image. 
After the collection process, I had 39228 number of data points. I then preprocessed this data by add lambda layer.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the error is almost stop declining at that epochs 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

####4. Final video result:
Here's a [link to my video result](track.mp4 )

