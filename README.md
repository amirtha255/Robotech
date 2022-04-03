# Robotech

Video - https://www.youtube.com/watch?v=4NkGZdtTPlA

## Motivation:
One of the biggest impediments to sustainability is trash. As the amount of trash ever grows with rising human consumption, the environment suffers as a result. Pollution, deforestation, and danger to wildlife are just some of the disastrous side effects. Hence there is an urgent need to handle the collection of trash, especially plastic and safely recycle it. 

However, could be very hazardous for us humans to handle trash material. It is very expensive to have humans clear litter in the streets, because of increasing wages for humans, and also the usage of human resources for a task that could have been handled by a robot. The human’s time is much better off contributing to the economy using his/her creativity and something which cannot be achieved using AI.


## Description:
  The robot needs to be able to clearly visually distinguish whether material is trash or not, and then get the exact dimensions of the material to determine the best grasping position. Trash can come in highly varying shapes, sizes, and colors, and hence detecting the presence of a rubbish object is not a trivial task. Also, since cleaning up is a highly repetitive task, there is huge scope for automation. It also could be grueling in terms of physical effort needed, so it could be done much better by a robot. The robot scans each object in its environment indicates its confidence level that the object detected is a piece of garbage, and that inference is visually depicted using different colored LED lights

## Implementation: 
  Instance Segmentation associates each pixel of the image with an instance label which is used to identify the objects from the background and also mark the exact boundary of each instance of the object present in the image. Faster R-CNN (Region-based Convolutional neural network) is a commonly used neural network architecture used to predict the bounding boxes of the objects and their labels. Mask R-CNN which extends Faster R-CNN by having another branch that predicts the object mask.
Here, we use Mask R-CNN to perform instance segmentation to analyze the image obtained from the camera mounted on the robot to identify the pixels in the image that contain trash. 

  Zero-shot learning is gaining popularity due to its generalizability. Usually, the model is pre-trained on large state-of-the-art datasets. The model predicts unseen data without any fine-tuning. This saves computational time and resources. It is also difficult to gather large amounts of labeled data, especially for tasks like trash detection and segmentation where the heavy annotation is involved. Zero-shot learning has proven to perform better than trained models through research like Open.ai’s CLIP. We observe that the pre-trained Mask R-CNN model with a ResNet-50 backbone detected bounding boxes and produced heatmaps significantly well for TACO dataset images with zero-shot capabilities. This is attributed to the ability of the model to generalize on out-of-domain data with prior generic knowledge. 



## References
1. Proença, P. F., & Simões, P. (2020). TACO: Trash annotations in context for litter detection. arXiv preprint arXiv:2003.06975.
2. https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
3. Zhang, J., Li, M., Feng, Y., & Yang, C. (2020). Robotic grasp detection based on image processing and random forest. Multimedia Tools and Applications, 79(3), 2427-2446.
4. https://learnopencv.com/mask-r-cnn-instance-segmentation-with-pytorch/
5. Majchrowska, S., Mikołajczyk, A., Ferlin, M., Klawikowska, Z., Plantykow, M. A., Kwasigroch, A., & Majek, K. (2022). Deep learning-based waste detection in natural and urban environments. Waste Management, 138, 274-284.
