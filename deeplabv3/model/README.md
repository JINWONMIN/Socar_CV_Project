### DeepLabv3 Model

Torchvision has pre-trained models available and I shall be using one of those models. <br>
Output channels are number of classes.

1. First, I get the pre-trained model using the <br>
   models.segmentation.deeplabv3_resnet101 method that downloads the pre-trained model into our system cache. Note resnet101 is the backbone for the deeplabv3 model obtained from this particular method. This decides the feature vector length that is passed onto the classifier.
   
2. The second step is the major step of modifying the segmentation head i.e. the classifier. This classifier is the part of the network and is responsible for vreating the final segmentation output. The change is done by replacing the classifier module of the model with a new DeepLabHead with a new number of output channels. 2048 is the feature vector size from the resnet101 backbone. (If decide to use another backbone, change this value accordingly.)

3. Finall, I set the model is set to train mode. This step is optional since can also do this in the training logic.
