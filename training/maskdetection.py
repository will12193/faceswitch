from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

def maskDetection(input_shape=(96, 96, 3)):
    input_image = Input(shape=input_shape)

    # 1st Conv layer
    model = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape)(input_image)
    model = MaxPooling2D((2, 2),padding='same')(model)

    # 2nd Conv layer
    model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
    model = MaxPooling2D((2, 2),padding='same')(model)

    # 3rd Conv layer
    model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
    model = MaxPooling2D((2, 2),padding='same')(model)

    # 4th Conv layer
    model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
    model = MaxPooling2D((2, 2),padding='same')(model)

    # 5th Conv layer
    model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
    model = MaxPooling2D((2, 2),padding='same')(model)

    # Fully connected layers
    model = Flatten()(model)

    #model = Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(model)
    model = Dense(1024)(model)
    #model = Dropout(0.2)(model)

    #model = Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(model)
    model = Dense(64)(model)
    #model = Dropout(0.2)(model)
    
    output= Dense(2, activation='softmax')(model)

    model = Model(inputs=[input_image], outputs=[output])
    return model