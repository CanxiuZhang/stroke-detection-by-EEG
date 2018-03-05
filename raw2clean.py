import numpy as np
import ArtRej
from tempfile import TemporaryFile
import featureExt
### loading data
# healthData = np.load('healthyRawData.npy')
# patientData = np.load('patientRawData.npy')
# healthData = np.load('healthyRawRawData.npy')
# patientData = np.load('patientRawRawData.npy')
healthData = np.load('healthTrain1.npy')
patientData = np.load('patientTrain1.npy')
# healthData = np.load('healthTest1.npy')
# patientData = np.load('patientTest1.npy')

print("health raw data shape", healthData.shape)
print("patient raw data shape", patientData.shape)

### artifact rejection
healthy_R = ArtRej.artifact_rejection(healthData)
patient_R = ArtRej.artifact_rejection(patientData)

healthy_AR = TemporaryFile()
np.save('healthy_AR.npy', healthy_R)
patient_AR = TemporaryFile()
np.save('patient_AR.npy', patient_R)

### split into 30 seconds epoch
healthy_R = np.load('healthy_AR.npy')
patient_R = np.load('patient_AR.npy')
#healthy_R = np.array(healthy_R).reshape(4,-1)
#patient_R = np.array(patient_R).reshape(4,-1)
epoch = 30
Fs = 256
observationNum_H = int(healthy_R.shape[1])//int(epoch*Fs)
observationNum_P = int(patient_R.shape[1])//int(epoch*Fs)
healthClean = np.array_split(healthy_R[:,0:int(observationNum_H*epoch*Fs-1)], observationNum_H, axis=1)
patientClean = np.array_split(patient_R[:,0:int(observationNum_P*epoch*Fs-1)], observationNum_P, axis=1)

### Summary
print("health data length after AR", healthy_R.shape[1])
print("patient data length after AR", patient_R.shape[1])
print("health observation number", observationNum_H)
print("patient observation number", observationNum_P)
print("health clean data shape", healthClean[0].shape)
print("patient clean data shape", patientClean[0].shape)
print("patient clean data patientClean[0]", patientClean[0])

### list to ndarray
lenth = epoch * Fs
healthCleanArray = np.zeros((observationNum_H, 4, lenth))
patientCleanArray = np.zeros((observationNum_P, 4, lenth))

for i in range(observationNum_H-1):
    for j in range(4):
        for k in range(0,lenth):
            healthCleanArray[i, j, k] = healthClean[i][j, k]

for i in range(observationNum_P-1):
    for j in range(4):
        for k in range(lenth):
            patientCleanArray[i, j, k] = patientClean[i][j, k]

print("shape of healthCleanArray", healthCleanArray[0,:,:].shape)
print("healthCleanArray", healthCleanArray[0,:,:])

### Save epoch
# Rawhealthy_clean = TemporaryFile()
# np.save('Rawhealthy_clean.npy', healthCleanArray)
# Rawpatient_clean = TemporaryFile()
# np.save('Rawpatient_clean.npy', patientCleanArray)

# healthy_clean = TemporaryFile()
# np.save('healthy_clean.npy', healthCleanArray)
# patient_clean = TemporaryFile()
# np.save('patient_clean.npy', patientCleanArray)

health_train_clean = TemporaryFile()
np.save('health_train_clean1.npy', healthCleanArray)
patient_train_clean = TemporaryFile()
np.save('patient_train_clean1.npy', patientCleanArray)

# health_test_clean = TemporaryFile()
# np.save('health_test_clean1.npy', healthCleanArray)
# patient_test_clean = TemporaryFile()
# np.save('patient_test_clean1.npy', patientCleanArray)

print("type of healthClean", type(healthCleanArray))
print("type of patientClean", type(patientCleanArray))
print("shape of healthClean", healthCleanArray.shape)
print("shape of patientClean", patientCleanArray.shape)

### Feature Extraction
featureData = []
for i in range(observationNum_H-1):
    featureData.append(featureExt.feature_extraction(healthClean[i]))
    featureData[i] = np.append(featureData[i], 0)
    print('healthy', i)

for i in range(observationNum_P-1):
    featureData.append(featureExt.feature_extraction(patientClean[i]))
    featureData[i+observationNum_H-1] = np.append(featureData[i+observationNum_H-1], 1)
    print('patient', i)

print("feature data length", len(featureData))
print("feature vector shape", featureData[0].shape)
print("health is", featureData[0][18])
print("patient is", featureData[observationNum_H][18])

# RawfeatureData = TemporaryFile()
# np.save('RawfeatureData.npy', featureData)

# FD = TemporaryFile()
# np.save('featureData.npy', featureData)

TrainFeature = TemporaryFile()
np.save('TrainFeature1.npy', featureData) # separate subject train feature set

# TestFeature = TemporaryFile()
# np.save('TestFeature1.npy', featureData) # separate subject test feature set
