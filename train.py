#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pipx install -U torch sahi ultralytics

get_ipython().system('pip install -U torch sahi ultralytics')


# In[4]:


import os


# In[5]:


from sahi.utils.yolov8 import (
    download_yolov8s_model
)

from sahi import AutoDetectionModel
import torch
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.predict import visualize_object_predictions
from IPython.display import Image
from numpy import asarray
from ultralytics import YOLO


# In[6]:


yolov8_model_path = "model/bestModel.pt"
download_yolov8s_model(yolov8_model_path)


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[8]:


detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.2,
    device=device, # or 'cuda:0'
)


# In[13]:


# import cv2
# from matplotlib import pyplot as plt

# # Load an image
image_path = 'images_waldo/waldo_20.jpg'
# image = cv2.imread(image_path)

# # Perform detection
# results = model(image)

# # Plot the results
# annotated_image = results[0].plot()
# plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
# plt.show()


# In[14]:


result = get_prediction(image_path, detection_model)


# In[15]:


result.export_visuals(export_dir="demo_data/")

Image("demo_data/prediction_visual.png")


# In[17]:


result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)


# In[18]:


result.export_visuals(export_dir="demo_data/")

Image("demo_data/prediction_visual.png")

