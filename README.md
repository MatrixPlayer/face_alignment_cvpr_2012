### Abstract
The provided project is my own version of Matthias Dantone et al. code based on the CVPR 2012 paper:

Dantone M, Fanelli G, Gall J. and Van Gool L., 
Real Time Facial Feature Detection using Conditional Regression Forest, IEEE Conference on Computer Vision and Pattern Recognition (CVPR'12), 2012.

This package contains the source code for training and evaluation of the 
Conditional Regression Forest. Additional to the source code you can find pretrained trees for head pose estimation and also for facial feature detection. 

### Building
This framework needs the open source computer vision library OpenCV 2.4.9 and Boost 1.55.0

### Demo Application
Running the demo application using the pretrained trees is easy.
```
./demo mode (0=training, 1=evaluate)
```
<p align="center">
  <img src="http://blog.gimiatlicho.webfactional.com/wp-content/uploads/2012/06/result_web.jpg" alt="Alignment"/>
</p>
