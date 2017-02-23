#**Behavioral Cloning** 

##Project No3 for 1st. Term of Self-Driving Car Nano Degree at Udacity

---
**Behavioral Cloning Project**

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior. To ahive this task, I only used only Udacity's [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)  and image augmentation.
* Build, a convolution neural network in Keras that predicts steering angles from images. I used [NVIDIA MODEL](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

###Model Architecture and Training Strategy

####1. Collecting the Data
I only used only Udacity's [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)  and image augmentation.


Images captured from the Udacity simulator:
* [Linx] (https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip)
* [MacOS] (https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip)
* [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip)

##### a.Image processing:
###### i. Select only the data with throttle > 0.2, this because contains more accurate data about what is the expected reaction for a car that is going forward.
###### ii. Select area of interest, the dimensions of the image is 320x160 pixeles. The area of interes is only the road, then I cut the image to get only the road.
###### iii. Resize the image to 200x66  pixeles and generate a numpy array of float values `img =np.asarray(img, dtype=np.float32)` This will be the data entry for the convoutional neuronal network.
It is returning the following image:

##### b. Image aumentation consist in the following functions:
###### i. Flip the image to have more steering angles to analyze. It involves to multiply the steering*-1
###### ii. Use right and left cameras. The steering angle would be less than the steering angle from the center camera. From the right camera's perspective, the steering angle would be larger than the angle from the center camera. `stering=stering +/- delta` I defined this delta as 0.25 (This data was obtained after evaluation on the pixels of images)
###### iii. Change brightness in order to get a more rebust model to the brightness.
###### iv. Image traslation to generate more images with steering variations.
###### v. Little perturbation in the steering, this because the the same image could have a little variation on the steering that will be a good answer, even for the humans we perform differents variations of the steering in the same situation. `nsteering= steering*(1+np.random.uniform(-1, 1)/50)`


#### 2. NVIDIA Model .
I used [NVIDIA MODEL](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)  :
My model consists has as an imput a RGB (3 filters) image with 200x66 as dimensions, then the imput is (200,66,3 )

*The data is normalized in the model using a Keras lambda layer (code in funtion get_NVIDIA_model). This is useful to have a more familiar range to work on. The reason is because in the process that we will train our network.
* I used ELU (Exponential Linear Units)  as activation funtion. The “exponential linear unit” (ELU) which speeds up learning in
deep neural networks and leads to higher classification accuracies. [Reference] (https://arxiv.org/pdf/1511.07289v1.pdf)
* I applied Dropout to the imput (code in funtion get_NVIDIA_model).  Dropout consists in randomly setting a fraction p of input units to 0 at each update during training time, which helps prevent overfitting. in this case `p=0.5`

* In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The validation set is 1/8 of the training set.

* The model was trained and validated on Udacity's [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)  with Image aumentation to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. However I have to modify the drive behavior to get this result.
Here is a visualization of the architecture:
