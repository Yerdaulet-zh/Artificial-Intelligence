Choosing an algorithm for object detection 

 

In the field of object detection, there are various architectures of artificial intelligence, all of them are used in specific tasks, most often the Mean Average Precision (mAP) metric is used to evaluate the performance of such algorithms. According to studies conducted on drug identification using YOLOv3, Faster R-CNN, and SSD algorithms [1], the Faster R-CNN algorithm showed the best result in terms of the mAP metric of 87.69%, but the main drawback is the calculation speed is only below 10 frames per second thus, this algorithm is not effective for tasks that require fast predictions, the SSD algorithm showed an average result for both criteria, for mAP 82.41%, for FPS just above 30 frames per second, but this is also not enough, YOLOv3 algorithm for mAP 80.17%, and in terms of FPS above 50 frames per second, this algorithm is inferior to everyone inaccuracy, but it calculates much faster, it is because of the calculation speed that many computer vision engineers choose the YOLO family of algorithms, since we have the simplest task of determining the face of people sitting in front of the screen, we chose the latter version of the algorithms in the YOLO family. 

 

Figure 1 - comparison of results of architectures 

 

The YOLO family is the most popular, fast-growing for real-time object detection, using only one neural grid, you can instantly get localized objects, since we only once run through the neurons without line layers, due to which the size and complexity of the calculation are much reduced and this greatly speeds up the work algorithm, this algorithm allows not only to determine, but can also classify objects, but in our case, we only determine the face. 

 

Table 1 - Results of different models from MS COCO data 

Model 

size 
(pixels) 

mAPval 
0.5:0.95 

mAPval 
0.5 

Speed 
CPU b1 
(ms) 

Speed 
V100 b1 
(ms) 

Speed 
V100 b32 
(ms) 

params 
(M) 

FLOPs 
@640 (B) 

YOLOv5n 

640 

28.0 

45.7 

45 

6.3 

0.6 

1.9 

4.5 

YOLOv5s 

640 

37.4 

56.8 

98 

6.4 

0.9 

7.2 

16.5 

YOLOv5m 

640 

45.4 

64.1 

224 

8.2 

1.7 

21.2 

49.0 

YOLOv5l 

640 

49.0 

67.3 

430 

10.1 

2.7 

46.5 

109.1 

YOLOv5x 

640 

50.7 

68.9 

766 

12.1 

4.8 

86.7 

205.7 

YOLOv5n6 

1280 

36.0 

54.4 

153 

8.1 

2.1 

3.2 

4.6 

YOLOv5s6 

1280 

44.8 

63.7 

385 

8.2 

3.6 

12.6 

16.8 

YOLOv5m6 

1280 

51.3 

69.3 

887 

11.1 

6.8 

35.7 

50.0 

YOLOv5l6 

1280 

53.7 

71.3 

1784 

15.8 

10.5 

76.8 

111.4 

YOLOv5x6 
﷟HYPERLINK "https://github.com/ultralytics/yolov5/releases"+ TTA 

1280 
1536 

55.0 
55.8 

72.7 
72.7 

3136 
- 

26.2 
- 

19.4 
- 

140.7 
- 

209.8 
- 
 

The latest architectures from the YOLO family are version 5, however, there are different variants among the fifth version of YOLO, as shown in table 1, they differ in speed and in the results of the COCO AP val metric from the MS COCO data, the most optimal architecture structure, which does not deviate so much from others in terms of quality and speed, is YOLOv5s. This architecture will be trained and used to determine a person's face. 

 

YOLOv5s architecture 

Figure 3 - YOLOv5s architecture  

Note - compiled by the author of the article [3] 

 

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

 

 
Figure 5 – Depiction of the task of object detection 

Note – Figure 4 and 5 are compiled by the author of the article [4] 

 

For instance, according to figure 5, where the green is the perfect right rectangle, and the second blue rectangle is the predicted rectangle from the model, the task is to determine how much the prediction is accurate. According to figure 4, we take over intersected areas and divide the number of regions and get values from 0 to 1, the higher IoU value the better prediction of the model. 

 

Non-maximum suppression 

 

YOLOv5s model predicts 50400 different rectangles to one image, so we need to localize which object where the most unique rectangle is located for localization, and it is not important not to miss close objects to solve this problem We have applied Non-Maximum Suppression technology that returns Localized objects. 

Идет вставка изображения... 
Figure 6 – Depiction of the task of non-maximum suppression 

Note - compiled by the author of the article [5] 

 

Let's solve the problem in Figure 6 with the help of NMS technology, where you need to determine whether there is only one object in the images, according to NMS, first filter by the confidence score of the probability of finding the object in the cell, then sort in descending order, after that, we take the highest for each class confidence score and its bounding box in order to determine whether other cells indicate the presence of another object, and for this, we first calculate the IoU value, then according to the IoU threshold, a decision is made, if the IoU threshold is 0.5 but we have the IoU difference below the trash hold, then we return this rectangle, as a localized object of this class. 

 

Training YOLOv5s network 

 

For training the YOLO algorithm, I used the official documentation of the algorithm on GitHub, the documentation includes various functions and features that are very convenient.  

 
Figure 7 - Yaml file content structure 


 

After installing the libraries, all images and their labels should be saved, as shown in Figure 7, where the path parameter specifies the main folder, the training parameter specifies the path to the training data, the val parameter specifies the path to the validation data, the test parameter specifies the path to the test data, parameter nc means the number of classes, the last parameter considers all the names of classes, all these settings should be saved in yamlfile.   

Figure 8 - command to start training 

Note - compiled by the author based on sources [3] 

 
	According to the command shown in figure 8, training is carried out by specifying the train.py file as the first parameter, then using --img 640 we specify the size of the input image, --batch we specify the batch size 16, --data we specify our yaml file, as shown in picture 10, --weights specify the YOLOv5 subversion or you can specify the path to the trained weights, but in our case, we specify the names of the yolov5s subversion, cache --cache so that each time you do not check whether the labels meet the requirements, and you can additionally specify --workers to allocate virtual threads to execute this command. 

 

YOLO dataset 

 

There are many different ready-made data in the public domain for training face recognition algorithms, but the quality of datasets leaves much to be desired, therefore, independently collecting various images from the Internet, and also obtaining images from other datasets that are not intended for training in human face recognition, through the site makesense.ai manually prepared, added some images from the augmentations received a total of 8689 images for training, and these images were sufficient for training. 

 

Data Preparation 

 

There are various types of annotations for learning the YOLO algorithm, but we use the popular version of annotation called PASCAL VOC XML, which is an open resource if you need to prepare annotations yourself, you can easily find this particular format. 

Идет вставка изображения... 
Figure 9 - an example of preparing an annotation for training 



 

Figure 9 is a good demonstration of PASCAL VOC XML formatting, all coordinates: x, y, width, and the height of the rectangle are normalized to image size, and central coordinates are defined. 

 

YOLOv5s training results 

 

Figure 10 - custom YOLOv5s training results 
 

There are three main model errors, the class error stands out very strongly, which is always equal to zero, which means that we have only one class, and the model cannot be mistaken because there is only one class and this is the face class, and also pay attention to the fact that that the validation error of the object gradually begins to grow, but very slowly, the results of precision, recall metrics are very good, they are all higher than 0.92%, mAP metrics also show good results of the model according to validation data. 
