#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install opendatasets


# In[2]:


#import opendatasets as od 


# In[3]:


#datasets="https://kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset/dataset"


# In[4]:


#od.download(datasets)


# In[5]:


import tensorflow as tf 
from tensorflow.keras.layers import Dense,Flatten,Rescaling,MaxPool2D,Conv2D,Dropout
from tensorflow.keras import Sequential
from tensorflow import keras 

import matplotlib.pyplot as plt 
import numpy as np 


# In[6]:


import os


# In[7]:


data_dir=os.path.join('C:\\Users\\ASUS TUF F15\\Projects\\Dogs classification\\dog-breed-image-dataset\\dataset')


# In[8]:


os.listdir(data_dir)


# In[9]:


os.listdir('C:\\Users\\ASUS TUF F15\\Projects\\Dogs classification\\dog-breed-image-dataset\\dataset\\Boxer')


# In[10]:


import PIL 


# In[11]:


PIL.Image.open('C:\\Users\\ASUS TUF F15\\Projects\\Dogs classification\\dog-breed-image-dataset\\dataset\\Boxer\\Boxer_20.jpg')


# In[12]:


batch_size=30
img_height=140
img_width=140


# In[13]:


train_ds=tf.keras.utils.image_dataset_from_directory(data_dir,
                                                    subset='training',
                                                    validation_split=0.2, 
                                                    seed=123, 
                                                    batch_size=batch_size, 
                                                    image_size=(img_width,img_height))


# In[14]:


val_ds=tf.keras.utils.image_dataset_from_directory(data_dir, 
                                                  subset='validation', 
                                                  validation_split=0.2, 
                                                  seed=123, 
                                                  batch_size=batch_size,
                                                  image_size=(img_width,img_height))


# In[15]:


class_name=train_ds.class_names


# In[16]:


class_name


# In[17]:


sample_img,labels=next(iter(train_ds))


# In[18]:


sample_img.shape


# In[19]:


labels


# In[20]:


labels.shape


# In[21]:


sample_img[0].numpy().astype('uint8')


# In[22]:


plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1): 
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_name[labels[i]])
        plt.axis('off')


# In[23]:


for image_batch,label_batch in train_ds: 
    print(image_batch.shape)
    print(label_batch.shape)
    break


# In[24]:


Autotune=tf.data.AUTOTUNE
train_ds=train_ds.shuffle(300).prefetch(buffer_size=Autotune)
val_ds=val_ds.prefetch(buffer_size=Autotune)


# In[25]:


num_classes=len(class_name)


# In[26]:


num_classes


# In[27]:


data_augmentation=Sequential([
    tf.keras.layers.RandomFlip('vertical',input_shape=(140,140,3)),
    tf.keras.layers.RandomRotation(0.2),
])


# In[28]:


model=Sequential([
    data_augmentation, 
    Rescaling(1./255), 
    Conv2D(16,3,padding='same',activation='relu'), 
    MaxPool2D(), 
    Conv2D(32,3,padding='same',activation='relu'), 
    MaxPool2D(), 
    Conv2D(64,3,padding='same',activation='relu'), 
    MaxPool2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128,activation='relu'), 
    Dense(10,activation='softmax')    
    ])
model.summary()


# In[30]:


blr=0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=blr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='none'),
              metrics=['accuracy'])


# In[31]:


epochs=30
history=model.fit(train_ds,validation_data=val_ds, 
                 epochs=epochs)


# In[32]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(4, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[33]:


class_name


# In[34]:


from tensorflow.keras.utils import load_img,img_to_array


# In[35]:


g_s=load_img('images.jpeg',target_size=(140,140))


# In[36]:


g_s


# In[37]:


img_arr=img_to_array(g_s)/255.0


# In[38]:


img_arr


# In[39]:


img_arr=img_arr.reshape(1,140,140,3)


# In[40]:


img_arr.shape


# In[41]:


predict=model.predict(img_arr).round(3)
predict


# In[42]:


np.argmax(model.predict(img_arr).round(3))


# In[43]:


class_name[np.argmax(model.predict(img_arr))]


# In[44]:


model.save('dog_breed_model.h5')


# In[ ]:





# In[ ]:




