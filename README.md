# Handwritten_Digit_Recognization

#########################################################
#   Name: Elias Zarate
#   Date: 12/10/2020
#
#########################################################

Requirements:

1. The following project leverages resources methods to digitize images to a 28x28 format (see image_resize() function in program). Handwritten images can be seen in the 'Handwritten_samples' folder and resized images can be seen in the 'Handwritten_resize' folder.

2. A CNN method was constructed using the referenced website to recognize all newly converted images. The accuracy of the model with the hand written digits were then displayed in the terminal.

Results:

1. Model evaluation
        Test loss: 0.039785899221897125
        Test accuracy: 0.9890999794006348
2. Accuracy of handwritten digits:
        accuracy: 20%

Explanations:
The model is not entirely accurate in predicting the handwritten digits. In fact the accuracy of the model differs from computation to computation which is pretty interesting. Some of the possible reasons for such high inaccuracies who predicting the handwritten digits may include the background color of which the digit was drawn, or the unorthodox style of the handwriting.


References:
https://www.sitepoint.com/keras-digit-recognition-tutorial/
