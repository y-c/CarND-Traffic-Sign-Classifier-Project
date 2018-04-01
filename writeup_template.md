# **Build a Traffic Sign Recognition Project**
---

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./hist.png "Dataset visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./13_yield.jpg "Traffic Sign 1"
[image5]: ./38_keep_right.jpg "Traffic Sign 2"
[image6]: ./01_speed_limit_30.jpg "Traffic Sign 3"
[image7]: ./17_no_entry.jpg "Traffic Sign 4"
[image8]: ./14_stop.jpg "Traffic Sign 5"

---

You're reading it! and here is a link to my [project code](https://github.com/y-c/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. These are the histograms of the training, validation, and testing data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

First, I used OpenCV to convert the images to grayscale because this can reduce the computation overhead and speed up the training process significantly. The reason why we can do this is, color does not carry important information since the traffic signs are designed in such way that even color blind people can tell.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Second, I normalized the image data to get all the data on the same scale: if the scales for different features are wildly different, this can have a knock-on effect on the ability to learn. Ensuring standardised feature values implicitly weights all features equally in their representation.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image  
| Convolution 5x5   | 1x1 stride, VALID padding, outputs 28x28x6 
| RELU					|						
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 
| Convolution 5x5	| 1x1 stride, VALID padding, outputs 10x10x16      
| RELU					|						
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 
| Flatten           | outputs 400
| Fully connected	| outputs 120      	
| Dropout           | Keep rate 0.75
| Fully connected	| outputs 84      	
| Dropout           | Keep rate 0.75
| Fully connected	| outputs 43      	



#### 3. Describe how you trained your model. 

I trained the model on AWS EC2. 

Here are my final training parameters:

* EPOCHS = 20
* BATCH_SIZE = 128
* SIGMA = 0.1
* OPIMIZER: AdamOptimizer (learning rate = 0.001)


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:

* Training set accuracy of 98.1%
* Validation set accuracy of 94.2% 
* Test set accuracy of 93.0%

I chose LeNet-5 DNN as the architecture. The reason is that it works well for image recognition for MNIST. My first try was a basic version of LeNet-5 that introduced in the class. I modified it to work with the input shape of 32x32x3. The validation accuracy was lower than 90%. Then I tried dropout and preprocessing. The result was pretty good. I also played with different parameters of EPOCHS. Training for more than 20 epochs did not seem to significantly improve the validation accuracy but increase the time. I also trained the network for 50 and more epochs, but I get a slightly decreasing accuracy. So I picked 20 as a tradeoff. I also played with different ways of processing the images into gray scale.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The "Keep Right" image might be difficult to classify because it was damaged. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield			
| Keep Right           | Keep Right
| 30 km/h	      		   | 100 km/h			
| No Entry             | No Entry 
| Stop Sign      		| Stop sign  

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the test accuaracy of 93%. I guess there are two reasons. First, the test data is not "clean", i.e., the sizes of the 5 new test images I found online are different from the training/validation/testing images, and they are even different from each other. Second, the model is not robust to perturbation. It needs more heterogeneous data to train.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

(1) Yield

![alt text][image4] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield   		
| 0.00     	          | No vehicles		
| 0.00     	          | Speed limit (60km/h)
| 0.00     	          | Ahead only
| 0.00     	          | No passing

This one is 100% accurate.

(2) Keep right

![alt text][image5] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999297           | Keep right   	
| 0.00     				| Road work		
| 0.00     				| General caution
| 0.00     				| Dangerous curve to the right
| 0.00     				| Yield


(3) Speed limit (30km/h)

![alt text][image6] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.49         			| Speed limit (100km/h)   
| 0.22         			| Speed limit (120km/h)
| 0.13         			| Speed limit (30km/h)
| 0.11         			| Speed limit (70km/h)
| 0.04         			| Speed limit (80km/h)


(4) No entry

![alt text][image7] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999714         	| No entry   
| 0.00         			| Turn right ahead
| 0.00         			| Turn left ahead
| 0.00         			| Beware of ice/snow
| 0.00         			| No passing


(5) Stop

![alt text][image8] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         	       | Stop   
| 0.00      			    | Keep right
| 0.00         			| Speed limit (30km/h)
| 0.00         			| Road work
| 0.00         			| No entry




