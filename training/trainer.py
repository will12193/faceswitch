import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
from imutils import paths
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from maskDetection import maskDetection
from resNet50 import resNet50

modelName = "resNet50_1"
# modelName = "maskdetection_1"

dataset = "./dataset"

# Label and setup dataset
imagePaths=list(paths.list_images(dataset))
data=[]
labels=[]

for i in imagePaths:
    label=i.split(os.path.sep)[-2]
    labels.append(label)
    image = load_img(i,target_size=(96,96))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)

data = np.array(data,dtype='float32')
labels = np.array(labels)

lb=LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split data
print('Split data')

train_X, test_X, train_Y, test_Y = train_test_split(
            data, labels, test_size=0.20, random_state=10, stratify=labels)

aug=ImageDataGenerator(
                       rotation_range=20,
                       zoom_range=0.15,width_shift_range=0.2,
                       height_shift_range=0.2,shear_range=0.15,
                       horizontal_flip=True,
                       vertical_flip=True,
                       fill_mode='nearest'
                       )

# Build the model
print('Build Model')

# model = maskdetection(input_shape=(96, 96, 3))
model = resNet50(input_shape=(96, 96, 3))

# Print model structure
model.summary()
print(model.summary())

# Define model learning rate, number of epochs, bach size, optimizer, and loss function
learningRate = 0.0005
epochs = 100
bachSize = 32
opt = Adam(learning_rate=learningRate,decay=learningRate/epochs)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

# Early stopping and model checkpoint
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
mc = ModelCheckpoint(os.path.join('../app/models', modelName+'.h5'), monitor='val_accuracy', mode='max', save_best_only=True)

# Fit the model
history = model.fit(
    aug.flow(train_X,train_Y,batch_size = bachSize),
    steps_per_epoch=len(train_X)//bachSize,
    validation_data=(test_X,test_Y),
    validation_steps=len(test_X)//bachSize,
    epochs=epochs,
    callbacks=[mc,es],
    verbose=1
)

# Print reports
try:
    predict = model.predict(test_X,batch_size=bachSize)
    predict = np.argmax(predict,axis=1)
    report = classification_report(test_Y.argmax(axis=1),predict,target_names=lb.classes_)
    print("Report:\n" + report)
except Exception as e: 
    print("Report failure: " + str(e))

try:
    scores = model.evaluate(test_X)
    print("Evaluate:\n" + str(scores))
except Exception as e: 
    print("Evaluate failure: " + str(e))

# Make training vs validation loss graphs
try:
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    plt.plot(loss_train, 'g', label='Training loss')
    plt.plot(loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(modelName+'-TrainingLoss.png')

    # Make training vs validation accuracy graphs
    acc_train = history.history['accuracy']
    acc_val = history.history['val_accuracy']
    plt.plot(acc_train, 'g', label='Training accuracy')
    plt.plot(acc_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig(modelName+'-TrainingAcc.png')

    # Plot loss and accuracy on the same graph
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()
    plt.savefig(modelName+'-LossAndAcc.png')
except Exception as e: 
    print("Plot failure: " + str(e))


"#<CITATION: USES SOME CODE FROM https://github.com/techyhoney/Facemask_Detection, Hiten Goyal>"