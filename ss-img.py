# Description: a program that classifies pixels in an image.Uses the ENet neural network architecture.
 
import cv2 
import numpy as np  
import os  
import imutils 
 
ORIG_IMG_FILE = 'test/test_image_4.jpg'
ENET_DIMENSIONS = (1024, 512) # dimensions that ENet was trained on
RESIZED_WIDTH = 600
IMG_NORM_RATIO = 1 / 255.0 
 
# Read the image
input_img = cv2.imread(ORIG_IMG_FILE)
 
# Resize the image while maintaining the aspect ratio
input_img = imutils.resize(input_img, width=RESIZED_WIDTH)
 
# Create a blob
# Preprocess the image to prepare it for deep learning classification
input_img_blob = cv2.dnn.blobFromImage(input_img, IMG_NORM_RATIO,
  ENET_DIMENSIONS, 0, swapRB=True, crop=False)
     
# Load the neural network (i.e. deep learning model)
print("Loading the neural network...")
enet_neural_network = cv2.dnn.readNet('./data/enet-cityscapes/enet-model.net')
 
# Set the input for the neural network
enet_neural_network.setInput(input_img_blob)
 
# Get the predicted probabilities for each of the classes
# These are the values in the output layer of the neural network
enet_neural_network_output = enet_neural_network.forward()
 
# Load the names of the classes
class_names = (
  open('./data/enet-cityscapes/enet-classes.txt').read().strip().split("\n"))
 
# Print out the shape of the output
#print(enet_neural_network_output.shape)
 
# Extract the key information about the ENet output
(number_of_classes, height, width) = enet_neural_network_output.shape[1:4] 
 
# Number of classes x height x width
#print(enet_neural_network_output[0])
 
# Find the class label that has the greatest probability for each image pixel
class_map = np.argmax(enet_neural_network_output[0], axis=0)
 
# Load a list of colors. Each class will have a unique color. 
if os.path.isfile('./data/enet-cityscapes/enet-colors.txt'):
  IMG_COLOR_LIST = (
    open('./data/enet-cityscapes/enet-colors.txt').read().strip().split("\n"))
  IMG_COLOR_LIST = [np.array(color.split(",")).astype(
    "int") for color in IMG_COLOR_LIST]
  IMG_COLOR_LIST = np.array(IMG_COLOR_LIST, dtype="uint8")
     
# If the list of colors file does not exist, we generate a random list of colors
else:
  np.random.seed(1)
  IMG_COLOR_LIST = np.random.randint(0, 255, size=(len(class_names) - 1, 3),
    dtype="uint8")
  IMG_COLOR_LIST = np.vstack([[0, 0, 0], IMG_COLOR_LIST]).astype("uint8")
   
# Tie each class ID to its color
# This mask contains the color for each pixel 
class_map_mask = IMG_COLOR_LIST[class_map]
 
# We now need to resize the class map mask so its dimensions is equivalent to the dimensions of the original image
class_map_mask = cv2.resize(class_map_mask, (
  input_img.shape[1], input_img.shape[0]),
    interpolation=cv2.INTER_NEAREST)
 
# Overlay the class map mask on top of the original image. We want the mask to be transparent
# We can do this by computing a weighted average of the original image and the class map mask
enet_neural_network_output = ((0.61 * class_map_mask) + (
  0.39 * input_img)).astype("uint8")
     
# Create a legend that shows the class and its corresponding color
class_legend = np.zeros(((len(class_names) * 25) + 25, 300, 3), dtype="uint8")
     
# Put the class labels and colors on the legend
for (i, (cl_name, cl_color)) in enumerate(zip(class_names, IMG_COLOR_LIST)):
  color_information = [int(color) for color in cl_color]
  cv2.putText(class_legend, cl_name, (5, (i * 25) + 17),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  cv2.rectangle(class_legend, (100, (i * 25)), (300, (i * 25) + 25),
                  tuple(color_information), -1)
 
# Combine the original image and the semantic segmentation image
combined_images = np.concatenate((input_img, enet_neural_network_output), axis=1) 
 
# Resize image if desired
#combined_images = imutils.resize(combined_images, width=1000)
 
# Display image
#cv2.imshow('Results', enet_neural_network_output) 
cv2.imshow('Results', combined_images) 
cv2.imshow("Class Legend", class_legend)
print(combined_images.shape)
cv2.waitKey(0) 
cv2.destroyAllWindows()