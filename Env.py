import gym
print(gym.__version__)
import cv2
import torch
from torchvision import transforms
import numpy as np

_mario_dress_r = 248
_mario_skin_r = 252
_mario_other_r = 172
def fill_white_col(image):
    """
    Replace white pixels in an RGB image with blue.
    
    Args:
        image: NumPy array of shape (height, width, 3), RGB image.
    
    Returns:
        color_image: NumPy array of shape (height, width, 3) with white pixels replaced by blue.
    """
    # Create a copy of the image to avoid modifying the original
    color_image = image.copy()
    
    # Define white pixel threshold (close to [255, 255, 255])
    # Adjust threshold for near-white pixels (e.g., 200 for leniency)
    white_threshold = 200
    white_mask = np.all(color_image >= [white_threshold, white_threshold, white_threshold], axis=2)
    
    # Replace white pixels with blue (RGB: [0, 0, 255])
    color_image[white_mask] = image[0,0,:]
    
    return color_image
def mark_mario_white(img):
    color_image = img.copy()

    # Extract Red channel
    red_channel = color_image[:, :, 0]

    # Create a mask where Red is 240, 252, or 172
    mask = (red_channel == _mario_dress_r)
    #mask = np.logical_or(mask, red_channel == _mario_skin_r)
    mask = np.logical_or(mask, red_channel == _mario_other_r)

    # Apply the mask to the image (change to white)
    color_image[mask] = [255, 255, 255]

    return color_image.copy()


def filter_white_black(img):
    color_image = img.copy()

    # Define white and black in RGB
    white = np.array([255, 255, 255])
    black = np.array([5, 5, 5])

    # Create masks
    is_white = np.all(color_image >= white, axis=2)
    is_black = np.all(color_image <= black, axis=2)

    # Everything else is "not white and not black"
    not_white_black = ~(is_white | is_black)
    # Set all "not white/black" pixels to gray
    color_image[not_white_black] = [128, 128, 128]  # gray color

    return color_image.copy()


def count_distinct_colors(img):
    # Reshape the image to (num_pixels, 3)
    new_img = img.copy()
    pixels = new_img.reshape(-1, 3)

    # Convert each pixel to a tuple (so it can be stored in a set)
    pixel_tuples = map(tuple, pixels)

    # Create a set of unique colors
    unique_colors = set(pixel_tuples)

    return len(unique_colors), unique_colors


def process(state):
    #mark_mario_white(fill_white_col(state[-200:-30,:,:]))
    image = filter_white_black(mark_mario_white(fill_white_col(state[-200:-30,:,:])))
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    # 歸一化到[0, 1]
    frame = frame / 255.0
    return frame.copy()
import cv2
import numpy as np
class State_Frame:
    def __init__(self):
        self.state_size = 84
        self.frame = None
        self.stacked_life =  None
        self.stacked_frame = None
    def get(self):
        #print(self.stacked_frame.shape)
        return self.stacked_frame.copy()
    def add_frame(self,frame):
        # 轉為灰度
        self.frame = process(frame)
        self.update()
        return
    def update(self):
        if self.stacked_frame is not None:
            self.stacked_frame = np.roll(self.stacked_frame,shift=-1,axis = 0)
            self.stacked_frame[-1,:,:] = self.frame
            #self.stacked_life = np.roll(self.stacked_life,shift=-1,axis=0)
            #self.stacked_life[-1] = self.life
        else:
            self.stacked_frame = np.stack([self.frame] * 4,axis = 0)
            #self.stacked_life = np.stack([self.life]*4,axis=0)
        return 