import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

modelName = "resNet50_1"
history=np.load(r'./resNet50_1-history.npy',allow_pickle='TRUE').item()

loss_train = history['loss']
loss_val = history['val_loss']
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
plt.savefig(modelName+'-TrainingLoss.png')
plt.clf()

# Make training vs validation accuracy graphs
acc_train = history['accuracy']
acc_val = history['val_accuracy']
plt.plot(acc_train, 'g', label='Training accuracy')
plt.plot(acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
plt.savefig(modelName+'-TrainingAcc.png')
plt.clf()

# Plot loss and accuracy on the same graph
pd.DataFrame(history).plot(figsize=(8,5))
plt.title('Training and Validation accuracy and loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
plt.savefig(modelName+'-LossAndAcc.png')