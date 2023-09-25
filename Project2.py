import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import cv2, os, time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

# instantiate the SIFT feature detector
sift = cv2.SIFT_create()

# 3.1 Use SIFT to find features and their descriptors in all the training set images from Project2_data/TrainingDataset/
keypoints = []
descriptors = []
y_train = []
x_train = []
for file in os.listdir('./Project2_data/TrainingDataset/'):
    # load the image and convert it to grayscale
    img = cv2.imread('./Project2_data/TrainingDataset/' + file, 0)
    # detect keypoints and extract descriptors
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)
    x_train.append(img)
    # butterfly images
    if file.startswith('024'):
        y_train.append(0)
    # hat images
    elif file.startswith('051'):
        y_train.append(1)
    # airplane images
    elif file.startswith('251'):
        y_train.append(2)

# test 3.1
imgKeyPoints1 = cv2.drawKeypoints(x_train[0], keypoints[0], None)
imgKeyPoints2 = cv2.drawKeypoints(x_train[50], keypoints[50], None)
imgKeyPoints3 = cv2.drawKeypoints(x_train[100], keypoints[100], None)

cv2.imshow('butterfly', imgKeyPoints1)
cv2.imshow('cowboyhat', imgKeyPoints2)
cv2.imshow('airplane', imgKeyPoints3)
cv2.waitKey(0)

# 3.2 Cluster all the SIFT feature descriptors from the training set into 100 clusters using k-means clustering
descriptors = np.concatenate(descriptors)
start = time.time()
dictionary = KMeans(n_clusters=100, init='k-means++', max_iter=100, n_init=1, random_state=0)
labels = dictionary.fit_predict(descriptors)
end = time.time()
print('Time taken to cluster (training): ', end - start)

# test 3.2
u_labels = np.unique(labels)
centers = dictionary.cluster_centers_
for i in u_labels:
    plt.scatter(descriptors[labels == i, 0], descriptors[labels == i, 1], label=i)
plt.scatter(centers[:, 0], centers[:, 1], s=80, c='black', marker='*')
plt.show()


# 3.3 Use the k-means clustering to create a histogram of the SIFT features for each image

def compute_histogram(image, sif, centroids):
    kp1, des1 = sif.detectAndCompute(image, None)
    hist1 = np.zeros(100)
    for d in des1:
        idx = np.argmin(np.linalg.norm(centroids - d, axis=1))
        hist1[idx] += 1
    return hist1 / len(kp1)


histograms = []

for img in x_train:
    hist = compute_histogram(img, sift, centers)
    histograms.append(hist)

# # test 3.3
for i in [0, 50, 100]:
    plt.bar(np.arange(100), histograms[i], align='center')
    plt.xlabel('Visual Words')
    plt.ylabel('Frequency')
    plt.title('Bag of Visual Words Histogram')
    plt.show()

# 3.4 Find the SIFT features and descriptors in the test images from Project2_data/TestDataset/. Assign these features
# to clusters. Create a normalized cluster histogram for each test image.

# Find SIFT features and descriptors in the test images
keypoints = []
descriptors = []
y_test = []
x_test = []
for file in os.listdir('./Project2_data/TestingDataset/'):
    # load the image and convert it to grayscale
    img = cv2.imread('./Project2_data/TestingDataset/' + file, 0)
    # detect keypoints and extract descriptors
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)
    x_test.append(img)
    # butterfly images
    if file.startswith('024'):
        y_test.append(0)
    # hat images
    elif file.startswith('051'):
        y_test.append(1)
    # airplane images
    elif file.startswith('251'):
        y_test.append(2)

# Assign features to clusters
descriptors = np.concatenate(descriptors)
labels = dictionary.predict(descriptors)

# Create a normalized cluster histogram for each test image
test_histograms = []

for img in x_test:
    hist = compute_histogram(img, sift, centers)
    test_histograms.append(hist)

# test 3.4
# test sift

imgKeyPoints1 = cv2.drawKeypoints(x_test[0], keypoints[0], None)
imgKeyPoints2 = cv2.drawKeypoints(x_test[10], keypoints[10], None)
imgKeyPoints3 = cv2.drawKeypoints(x_test[20], keypoints[20], None)

cv2.imshow('butterfly', imgKeyPoints1)
cv2.imshow('cowboyhat', imgKeyPoints2)
cv2.imshow('airplane', imgKeyPoints3)
cv2.waitKey(0)

# test kmeans
u_labels = np.unique(labels)
centers = dictionary.cluster_centers_
for i in u_labels:
    plt.scatter(descriptors[labels == i, 0], descriptors[labels == i, 1], label=i)
plt.scatter(centers[:, 0], centers[:, 1], s=80, c='black', marker='*')
plt.show()
# test histogram
for i in [0, 10, 20]:
    plt.bar(np.arange(100), test_histograms[i], align='center')
    plt.xlabel('Visual Words')
    plt.ylabel('Frequency')
    plt.title('Bag of Visual Words Histogram')
    plt.show()

# 3.5 Assign each test image (from TestingDataset) to one of the three classes by finding the class histogram that
# is closest to the test image histogram. Use SKLearn’s KNeighborsClassifier to classify the test images.
# Time the testing time and report the accuracy of your classifier on the test set.

# creating classifier
knn = KNeighborsClassifier(n_neighbors=1)
# time testing time
start = time.time()
knn.fit(histograms, y_train)
end = time.time()

# testing classifier
y_pred = knn.predict(test_histograms)
print('KNN Accuracy: ', accuracy_score(y_test, y_pred))
print('KNN Time: ', end - start)

# generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 3.6 Use a linear SVM to classify the test images.

# creating classifier
svm = SVC(kernel='linear')
# time testing time
start = time.time()
svm.fit(histograms, y_train)
end = time.time()

# testing classifier
y_pred = svm.predict(test_histograms)
print('SVM Accuracy: ', accuracy_score(y_test, y_pred))
print('SVM Time: ', end - start)

# generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 3.7 Use a kernel SVM to classify the test images.

# creating classifier
svm = SVC(kernel='rbf')
# time testing time
start = time.time()
svm.fit(histograms, y_train)
end = time.time()

# testing classifier
y_pred = svm.predict(test_histograms)
print('Kernel SVM Accuracy: ', accuracy_score(y_test, y_pred))
print('Kernel SVM Time: ', end - start)

# generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 4 create a plot that projects the historgrams associated with the training data onto the first three principal directions.

# create PCA object
pca = PCA(n_components=3)
# fit PCA
pca.fit(histograms)
# transform data
histograms_pca = np.dot(histograms, pca.components_.T)

# create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot data
for i in np.unique(y_train):
    ix = np.where(y_train == i)
    if i == 0:
        l = 'butterfly'
    elif i == 1:
        l = 'cowboy'
    else:
        l = 'airplane'
    ax.scatter(histograms_pca[ix, 0], histograms_pca[ix, 1], histograms_pca[ix, 2], label=l)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.show()