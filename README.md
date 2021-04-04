# BrainCancerDetection
This is a project that uses Convolutional Neural Networks, that is made using Tensorflow and Keras to classify 4 different types of Brain Cancers. They are glioma tumors,  meningioma tumors, no tumors, and pituitary tumors.

To improve my accuracy I did denoising of the image by using a gaussian blur, and then I contoured the image so that the model can see all the different parts of the image. I also did image contouring so that we can segment the actual brain mri and the other black parts of the image.

![braincancerclassifier-ipynb-Colaboratory (2)](https://user-images.githubusercontent.com/47342287/113519009-b5386f00-9557-11eb-96bd-069aca410bf2.png)
I thresholded some my images so that I we can better distingush the actual brain mri part and the other black parts around the mri scan.

![braincancerclassifier-ipynb-Colaboratory (3)](https://user-images.githubusercontent.com/47342287/113519054-f0d33900-9557-11eb-99e9-ebf93e8b0470.png)
This is my dataset but not it is contoured which segmented the different parts of the mri scan.

# Next Steps:

- Use the model for real-time predictions
- Experiment with different model architectures for optimal performance
- Use different techniques in order to be able to do special preprocessing so that this data is enhanced and the model can achieve greater accuracies
