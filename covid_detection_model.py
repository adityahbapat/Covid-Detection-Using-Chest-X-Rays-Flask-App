#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import shutil
import glob
import matplotlib.pyplot as plt


# In[2]:


covid_imgs = pd.read_excel("C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\COVID-19_Radiography_Dataset\\COVID.metadata.xlsx")


# In[3]:


opacity_images = pd.read_excel("C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\COVID-19_Radiography_Dataset\\Lung_Opacity.metadata.xlsx")


# In[4]:


normal_images = pd.read_excel("C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\COVID-19_Radiography_Dataset\\Normal.metadata.xlsx")


# In[5]:


pneumonia_images = pd.read_excel("C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\COVID-19_Radiography_Dataset\\Viral Pneumonia.metadata.xlsx")


# In[6]:


ROOT_DIR = "C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\COVID-19_Radiography_Dataset\\"
imgs = ['COVID','Lung_Opacity','Normal','Viral Pneumonia']

NEW_DIR = "C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\content\\all_images\\"


# In[7]:


train_path  = "C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\content\\all_images\\train_test_split\\train"
valid_path  = "C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\content\\all_images\\train_test_split\\validation\\"
test_path   = "C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\content\\all_images\\train_test_split\\test\\"


# In[8]:



import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model, model_from_json 
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D


# In[9]:


train_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input,
                                    zoom_range= 0.2,
                                    horizontal_flip= True,
                                    shear_range= 0.2,

                                    )

train = train_data_gen.flow_from_directory(directory= train_path,
                                          target_size=(224,224))


# In[10]:


validation_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input  )

valid = validation_data_gen.flow_from_directory(directory= valid_path,
                                                target_size=(224,224))


# In[11]:


test_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input )

test = train_data_gen.flow_from_directory(directory= test_path ,
                                          target_size=(224,224),
                                          shuffle= False)


# In[12]:


class_type = {0:'Covid',  1 : 'Normal'}


  # In[13]:

def fnc_cnn():
  from tensorflow.keras.applications.resnet50 import ResNet50
  from tensorflow.keras.layers import Flatten , Dense, Dropout , MaxPool2D


  # In[14]:

  res = ResNet50( input_shape=(224,224,3), include_top= False) # include_top will consider the new weights


  # In[15]:


  for layer in res.layers:           # Dont Train the parameters again
    layer.trainable = False


  # In[16]:


  x = Flatten()(res.output)
  x = Dense(units=2 , activation='sigmoid', name = 'predictions' )(x)

  # creating our model.
  model = Model(res.input, x)


  # In[17]:


  model.compile( optimizer= 'adam' , loss = 'categorical_crossentropy', metrics=['accuracy'])


  # In[18]:


  # implementing early stopping and model check point

  from keras.callbacks import EarlyStopping
  from keras.callbacks import ModelCheckpoint

  es = EarlyStopping(monitor= "val_accuracy" , min_delta= 0.01, patience= 3, verbose=1)
  mc = ModelCheckpoint(filepath="bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only= True)


  # In[19]:


  hist = model.fit(train, steps_per_epoch= 10, epochs= 30, validation_data= valid , validation_steps= 10, callbacks=[es,mc])


  # In[20]:


  ## load only the best model
  from keras.models import load_model
  model = load_model("bestmodel.h5")


  # serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")
  print("Saved model to disk")



# In[21]:


from tensorflow.keras.preprocessing import image

def get_img_array(img_path):
  """
  Input : Takes in image path as input
  Output : Gives out Pre-Processed image
  """
  path = img_path
  img = image.load_img(path, target_size=(224,224,3))
  img = image.img_to_array(img)
  img = np.expand_dims(img , axis= 0 )

  return img


# In[22]:


# path for that new image

def xray_test(path):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")



    img = get_img_array(path)
    res_list = []
    res = class_type[np.argmax(loaded_model.predict(img))]
    xray_type = f"The given X-Ray image is of type = {res}"
    res_list.append(xray_type)
    xray_covid = f"The chances of image being Covid is : {round(loaded_model.predict(img)[0][0]*100,4)} %"
    res_list.append(xray_covid)
    xray_normal = f"The chances of image being Normal is : {round(loaded_model.predict(img)[0][1]*100,4)} %"
    res_list.append(xray_normal)
    return res_list
    # plt.imshow(img[0]/255, cmap = "gray")
    # plt.title("input image")
    # plt.show()


# path = "C:\\Users\\Aditya\\Downloads\\Covid_detection_CNN\\content\\all_images\\COVID-1031.png"

#predictions: path:- provide any image from google or provide image from all image folder


# to display the image



# In[ ]:
