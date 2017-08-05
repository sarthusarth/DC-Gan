
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
from datetime import datetime


# In[47]:

import keras
from keras.models import Sequential
from keras.layers import Activation,Dense,Conv2D,Dropout,Flatten,BatchNormalization,Reshape,UpSampling2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop,Adam


# In[48]:

#ds=pd.read_csv('../data/mnist_data/train.csv')
ds=pd.read_csv('/input/train.csv')


# In[49]:

data=np.array(ds)


# In[50]:

X=data[:,1:]

X=(X/255.0)
# In[51]:

X=np.reshape(X,(X.shape[0],28,28,1))


# In[52]:

D=Sequential()
depth=64
dropout=0.3
D.add(Conv2D(64,5,strides=2,input_shape=(28,28,1),padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Conv2D(128,5,strides=2,padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Conv2D(256,5,strides=2,padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Conv2D(512,5,strides=1,padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Flatten())
D.add(Dense(1))
D.add(Activation('sigmoid'))
D.summary()


# In[53]:

G=Sequential()
dropout=0.3
G.add(Dense(12544,input_shape=(100,)))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(Reshape((7,7,256)))
G.add(Dropout(dropout))
G.add(UpSampling2D())
G.add(Conv2DTranspose(128,5,padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(UpSampling2D())
G.add(Conv2DTranspose(64,5,padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(Conv2DTranspose(32,5,padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(Conv2DTranspose(1,5,padding='same'))
G.add(Activation('sigmoid'))
G.summary()


# In[54]:

optimizer =Adam(lr=0.001, decay=1e-4)
D.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
D.trainable=False

# In[55]:

optimizer_G = Adam(lr=0.0005, decay=1e-4)
AM=Sequential()
AM.add(G)
AM.add(D)
AM.compile(loss='binary_crossentropy',optimizer=optimizer_G,metrics=['accuracy'])


# In[56]:

t1=datetime.now()
for epoch in range(5000):
    noise=np.random.uniform(-1,1,size=[256,100])
    images_fake=G.predict(noise)
    rand_no=random.randint(0,40000)
    x=np.concatenate((X[rand_no:rand_no+256],images_fake))
    y=np.ones([512,1])
    y[256:,:]=0
    D.trainable=True
    D.train_on_batch(x, y)
    D.trainable=False
    y = np.ones([256, 1])
    noise = np.random.uniform(-1.0, 1.0, size=[256, 100])
    a_loss=AM.train_on_batch(noise,y)


t2=datetime.now()


# In[57]:

#from keras.models import load_model
G.save('/output/G.h5')
#D.save('/output/D.h5')



#print str(t2-t1)[:7]

noise=np.random.uniform(-1,1,size=[100,100])
images_fake=G.predict(noise)
for ix in range(100):
    test=np.reshape(images_fake[ix],(28,28))
    #plt.imshow(test,cmap='gray')
    #plt.axis('off')
    #plt.show()
    name='/output/'+str(ix)+'.png'
    plt.imsave(name,test,cmap='gray')
