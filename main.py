from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

(x_train,y_train),(x_test,y_test) = mnist.load_data()

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4), input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

early_stop = EarlyStopping(monitor='val_loss',patience=2)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

#Normalize the data
x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

model.fit(x_train, y_cat_train, epochs = 10, validation_data=(x_test, y_cat_test),callbacks=[early_stop])

print("#########evaluate########")

model.evaluate(x_test,y_cat_test)


y_test_pred = model.predict(x_test)

y_test_pred_classes = np.argmax(y_test_pred,axis = 1)

print(classification_report(y_test,y_test_pred_classes))

# Save the model as SavedModel format
model.save('./output_model/mnist_digit_classifier.keras')
print("Model saved as 'mnist_digit_classifier'")

# Also save as .h5 for backup
model.save('./output_model/mnist_digit_classifier.h5')
print("Model also saved as 'mnist_digit_classifier.h5'")