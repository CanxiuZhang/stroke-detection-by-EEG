import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler

# raw_healthydata = np.load('Rawhealthy_clean.npy') # healthy epoch (117,4,7680)
# raw_patientdata = np.load('Rawpatient_clean.npy') # patient epoch (119,4,7680)

# healthydata = np.load('healthy_clean.npy') # healthy epoch (117,4,7680)
# patientdata = np.load('patient_clean.npy') # patient epoch (119,4,7680)

### seperate subject
raw_healthydata = np.load('health_test_clean.npy') # healthy epoch (117,4,7680)
raw_patientdata = np.load('patient_test_clean.npy') # patient epoch (119,4,7680)
healthydata = np.load('health_train_clean.npy') # healthy epoch (117,4,7680)
patientdata = np.load('patient_train_clean.npy') # patient epoch (119,4,7680)


X = []
y = []
for i in range(healthydata.shape[0]):
    X.append(healthydata[i])
    y.append(-1)
for i in range(patientdata.shape[0]):
    X.append(patientdata[i])
    y.append(1)

normalized_X = X
for i in range(len(X)):
    normalized_X[i] = preprocessing.normalize(X[i])
X_train = normalized_X
y_train = y

X_test = []
y_test = []
for i in range(raw_healthydata.shape[0]):
    X_test.append(raw_healthydata[i])
    y_test.append(-1)

for i in range(raw_patientdata.shape[0]):
    X_test.append(raw_patientdata[i])
    y_test.append(1)
normalized_X_test = X_test
for i in range(len(X_test)):
    normalized_X_test[i] = preprocessing.normalize(X_test[i])
X_test = normalized_X_test

### shuffle
#X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.3, random_state=42, shuffle=True)

# data pre-processing
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = X_train.reshape(-1, 1, 4, 7680)
X_test = X_test.reshape(-1, 1, 4, 7680)
y_train = np_utils.to_categorical(y_train, num_classes=2)  # Converts a class vector (integers) to binary class matrix.
y_test = np_utils.to_categorical(y_test, num_classes=2)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (40, 4, 7680)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 4, 7680),
    filters=40,
    kernel_size=(1, 25),
    strides=1,
    # padding='same',     # Padding method
    data_format='channels_first',
))

### activation
#model.add(Activation('elu'))

model.add(Convolution2D(
    filters=40,
    kernel_size=(4, 1),
    strides=1,
    #padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('elu'))

# Pooling layer 1 (max pooling) output shape (32, 4, 768)
model.add(AveragePooling2D(
    pool_size=(1, 200),
    strides=(1, 200),
    padding='same',    # Padding method
    data_format='channels_first',
))

#model.add(Activation('elu'))

# Fully connected layer 1 input shape (64 * 4 * 64) = (16384), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('elu'))

# Fully connected layer 2 to shape (2) for 2 classes
model.add(Dense(2))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
### epoch ??
model.fit(X_train, y_train, epochs=1, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print(model.summary())

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

plot_model(model, to_file='model.png')
