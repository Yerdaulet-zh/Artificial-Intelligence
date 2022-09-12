 

YOLOv5s architecture 

 

The architecture shown in Figure 3 consists of two backbones, head, and several components such as: Conv, Bottleneck (True), Bottleneck (False), BCSPn, Focus, SPP. 

The task of the Backbone is a combination of components that take place with CNN and pooling layers, the task of this part is to identify distinguishing features from the input image, in other words, image detail. 

The task of head is a combination of the distinguishing features of the input image and predict the bounding box, confidence, and class. 

The focus component increases the number of channels by copying the input image and at the end, concatenation is applied and transferred to the next layer, the next ultra-precise layer with batch normalization 2d, reveals new features, details images, increases the number of channels. The idea of a bottleneck component is very similar to looking at a residual connection that copies the input layer and after several layers concatenates the copied and received layer results, but in the bottleneck (True) we will add instead of adding. Bottleneck (False) does not copy or accept layer stacking, in our case, it will just apply two CNN layers. BCSPn of the component that copies the input layer and applies one CNN layer and concatenates with the result, to which the CNN layer was applied, after the bottleneck (True), at the end of the CNN layer, after the concatenations we apply batch normalization 2d, Leaky Relu is used as the activation function, and one CNN component at the output. The green BCSPn component is different than the normal bottleneck that uses the bottleneck (False). An SPP component that uses three pooling layers with different kernel sizes, one unused pooling layer that is instead concatenated at the end. 

 

Loss in YOLOv5 

 

It is possible to express the error in YOLOv5: 

			(1) 

Where: The final loss is the sum of the losses: boxloss, classloss, objectloss;  

Lbox - rectangle coordinate error; 

Lclass - predicted class error; 

Lobject - error finding an object in a cell; 

In our case, in defining just only one class, which is the face, we use to compute class loss popular binary cross-entropy loss as follows:   

		(2) 
 

For the box loss computation in yolov5 we use somehow similar loss to the SSR as follows: 

			(3) 
 

Confidence loss, called the loss that computes is there object or not as follows: 

	(4) 
 

YOLOv5s make predictions in three different dimensions, for example, the first scale prediction block with a large dimension of 80x80x6, the second block 40x40x6, the third block 20x20x6 for a total of 50400 predictions with the different bounding box, confidence, class score, in order to get correct predictions, a technology called non-maximum suppression, but before that, we need to use a technic called intersection over union (IoU). 

 

Intersection over union 

 

IoU is a method used to calculate the overlap of two rectangles, the higher the IoU, the greater the overlap region, the method widely used when you need to determine the perfect suitable object among hundreds of objects that the model predicts, and with help of this method we calculate mAP to understand how does the model fit well, also to avoid repeating rectangle prediction of one class. 

 
Figure 4 – Representation of IoU calculation 

 

 

For instance, according to figure 5, where the green is the perfect right rectangle, and the second blue rectangle is the predicted rectangle from the model, the task is to determine how much the prediction is accurate. According to figure 4, we take over intersected areas and divide the number of regions and get values from 0 to 1, the higher IoU value the better prediction of the model. 

 

Non-maximum suppression 

 

YOLOv5s model predicts 50400 different rectangles to one image, so we need to localize which object where the most unique rectangle is located for localization, and it is not important not to miss close objects to solve this problem We have applied Non-Maximum Suppression technology that returns Localized objects. 

Идет вставка изображения... 


 

Let's solve the problem in Figure 6 with the help of NMS technology, where you need to determine whether there is only one object in the images, according to NMS, first filter by the confidence score of the probability of finding the object in the cell, then sort in descending order, after that, we take the highest for each class confidence score and its bounding box in order to determine whether other cells indicate the presence of another object, and for this, we first calculate the IoU value, then according to the IoU threshold, a decision is made, if the IoU threshold is 0.5 but we have the IoU difference below the trash hold, then we return this rectangle, as a localized object of this class. 

 

Training YOLOv5s network 

 

For training the YOLO algorithm, I used the official documentation of the algorithm on GitHub, the documentation includes various functions and features that are very convenient.  

 


 

After installing the libraries, all images and their labels should be saved, as shown in Figure 7, where the path parameter specifies the main folder, the training parameter specifies the path to the training data, the val parameter specifies the path to the validation data, the test parameter specifies the path to the test data, parameter nc means the number of classes, the last parameter considers all the names of classes, all these settings should be saved in yamlfile.   

 
	According to the command shown in figure 8, training is carried out by specifying the train.py file as the first parameter, then using --img 640 we specify the size of the input image, --batch we specify the batch size 16, --data we specify our yaml file, as shown in picture 10, --weights specify the YOLOv5 subversion or you can specify the path to the trained weights, but in our case, we specify the names of the yolov5s subversion, cache --cache so that each time you do not check whether the labels meet the requirements, and you can additionally specify --workers to allocate virtual threads to execute this command. 

 

YOLO dataset 

 

There are many different ready-made data in the public domain for training face recognition algorithms, but the quality of datasets leaves much to be desired, therefore, independently collecting various images from the Internet, and also obtaining images from other datasets that are not intended for training in human face recognition, through the site makesense.ai manually prepared, added some images from the augmentations received a total of 8689 images for training, and these images were sufficient for training. 

 

Data Preparation 

 

There are various types of annotations for learning the YOLO algorithm, but we use the popular version of annotation called PASCAL VOC XML, which is an open resource if you need to prepare annotations yourself, you can easily find this particular format. 

Идет вставка изображения... 


 

A good demonstration of PASCAL VOC XML formatting, all coordinates: x, y, width, and the height of the rectangle are normalized to image size, and central coordinates are defined. 

 

YOLOv5s training results 

 

There are three main model errors, the class error stands out very strongly, which is always equal to zero, which means that we have only one class, and the model cannot be mistaken because there is only one class and this is the face class, and also pay attention to the fact that that the validation error of the object gradually begins to grow, but very slowly, the results of precision, recall metrics are very good, they are all higher than 0.92%, mAP metrics also show good results of the model according to validation data. 
