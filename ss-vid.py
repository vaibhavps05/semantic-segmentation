# Description: A program that classifies pixels in a video. Uses the ENet neural network architecture.
 
import cv2 
import numpy as np 
import os
import imutils 
 
filename = './data/test/car_detect_1.mp4'
file_size = (1920,1080) 
 
# We want to save the output to a video file
output_filename = 'semantic_seg_car_detect_1.mp4'
output_frames_per_second = 20.0
 
ENET_DIMENSIONS = (1024, 512) # Dimensions that ENet was trained on
RESIZED_WIDTH = 1200
IMG_NORM_RATIO = 1 / 255.0 
 
# Load the names of the classes
class_names = (
  open('./data/enet-cityscapes/enet-classes.txt').read().strip().split("\n"))
     
# Load a list of colors 
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
 
def main():
 
  # Load a video
  cap = cv2.VideoCapture(filename)
 
  # Create a VideoWriter object so we can save the video output
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  result = cv2.VideoWriter(output_filename,  
                           fourcc, 
                           output_frames_per_second, 
                           file_size) 
     
  # Process the video
  while cap.isOpened():
         
    # Capture one frame at a time
    success, frame = cap.read() 
         
    if success:
         
      # Resize the frame while maintaining the aspect ratio
      frame = imutils.resize(frame, width=RESIZED_WIDTH)
 
      # Create a blob
      # Preprocess the frame to prepare it for deep learning classification
      frame_blob = cv2.dnn.blobFromImage(frame, IMG_NORM_RATIO,
                     ENET_DIMENSIONS, 0, swapRB=True, crop=False)
     
      # Load the neural network 
      enet_neural_network = cv2.dnn.readNet('./data/enet-cityscapes/enet-model.net')
 
      # Set the input for the neural network
      enet_neural_network.setInput(frame_blob)
 
      # Get the predicted probabilities for each of the classes (e.g. car, sidewalk)
      # These are the values in the output layer of the neural network
      enet_neural_network_output = enet_neural_network.forward()
 
      # Extract the key information about the ENet output
      (number_of_classes, height, width) = (
        enet_neural_network_output.shape[1:4]) 
 
      # Find the class label that has the greatest probability for each frame pixel
      class_map = np.argmax(enet_neural_network_output[0], axis=0)
 
      # This mask contains the color for each pixel. 
      class_map_mask = IMG_COLOR_LIST[class_map]
 
      # We now need to resize the class map mask so its dimensions is equivalent to the dimensions of the original frame
      class_map_mask = cv2.resize(class_map_mask, (
        frame.shape[1], frame.shape[0]), 
        interpolation=cv2.INTER_NEAREST)
 
      # Overlay the class map mask on top of the original frame
      enet_neural_network_output = ((0.90 * class_map_mask) + (
        0.10 * frame)).astype("uint8")
     
      # Combine the original frame and the semantic segmentation frame
      combined_frames = np.concatenate(
        (frame, enet_neural_network_output), axis=1) 
 
      # Resize frame if desired
      combined_frames = imutils.resize(combined_frames, width=1920)
 
      # Create an appropriately-sized video frame. We want the video height to be 1080 pixels
      adjustment_for_height = 1080 - combined_frames.shape[0]
      adjustment_for_height = int(adjustment_for_height / 2)
      black_img_1 = np.zeros((adjustment_for_height, 1920, 3), dtype = "uint8")
      black_img_2 = np.zeros((adjustment_for_height, 1920, 3), dtype = "uint8")
 
      # Add black padding to the video frame on the top and bottom
      combined_frames = np.concatenate((black_img_1, combined_frames), axis=0) 
      combined_frames = np.concatenate((combined_frames, black_img_2), axis=0) 
       
      # Write the frame to the output video file
      result.write(combined_frames)
         
    # No more video frames left
    else:
      break
             
  # Stop when the video is finished
  cap.release()
     
  # Release the video recording
  result.release()
 
main()