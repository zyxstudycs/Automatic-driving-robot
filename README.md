# Automatic-driving-robot
Autonomous navigation for mobile robots is a popular application for deep learning, made ever more relevant by the development of the self driving car. Our project aims to teach robot run in a general case simulated traffic system, mainly by correcting itself when deviating off course and off lane, and to recognize traffic signs.

# Videos
1. video for following the lane:
2. video for left turn with 'turn left sign':
3. video for going straight without 'turn left sign':

# Robot
![alt text](https://github.com/zyxstudycs/Automatic-driving-robot/blob/master/image/robot.png)

# model for decision
1. CNN part: We used the pre-trained Inception Resnet V2 weights, extracted before the last fully connected layer, and added batch normalization, drop out with drop rate of 0.3 and a dense layer. So, we only re-train the last layer.To speed up our training process, we first pre-process our images into small arrays (we call it bottleneck result) by computing the layer before the “logits” layer, and save the result in a directory called ‘bottle_neck’. Then we actually re-train the last layer using the input of  “bottleneck” result.
We were able to get 81% accuracy in this part, it is able to follow the lanes.

2. To let the robot make more precise decisions when discover road sign, we feed the last layer of CNN in sequence into LSTM, and get much higher accuracy for turning in crossroads, but still get confused when discover traffic signs.

3. To make it successfully detect traffic signs, and make decision based on them, we add filter method on each image, aims to highlight traffic signs in the image, this make great progress on the accuracy and finally get 94% of accuracy.


