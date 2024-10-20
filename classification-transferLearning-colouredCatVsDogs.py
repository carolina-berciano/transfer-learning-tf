# import libraries
import tensorflow as tf
import matplotlib.pylab as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import numpy as np
import PIL.Image as Image

##################################################################################
##### Use a TensorFlow Hub model for prediction (Direct use, no fine-tuning) #####
##################################################################################
# find a tf2 compatible model (https://www.kaggle.com/models?query=tf2&task=16686&tfhub-redirect=true)
CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
# Find the image dimensions the model has been trained with (we will need to resize our image dataset for compatibility)
IMAGE_RES = 224

# Download the pre-tained model and create a Keras model from it
model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])

# get an unseen image from the internet and convert it to be compatible with model
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))
grace_hopper = np.array(grace_hopper)/255.0

# Predict image class
result = model.predict(grace_hopper[np.newaxis, ...])  # models always want a batch of images to process
predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

# download the ImageNet labels and fetch the row that the model predicted
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Check that the prediction class is correct
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

#############################################################################################
##### Use a TensorFlow Hub model for cats vs dogs datasets (Direct use, no fine-tuning) #####
#############################################################################################

# use TensorFlow Datasets to load the Dogs vs Cats dataset
(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    with_info=True,
    as_supervised=True,
    split=['train[:80%]', 'train[80%:]'],
)

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes


# reformat all images to the resolution expected by MobileNet (224, 224)
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


BATCH_SIZE = 32

train_batches = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

# predict the images in our Dogs vs. Cats dataset
image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

result_batch = model.predict(image_batch)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(predicted_class_names)

########################################################################################
##### Transfer Learning Fine-tune a TensorFlow Hub model for cats vs dogs datasets #####
########################################################################################

# create model from TensorFlow Hub model
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(
    URL,
    input_shape=(IMAGE_RES, IMAGE_RES,3))  # define model without classification layer (a.k.a features extraction layer)
feature_extractor.trainable = False  # freeze the variables in the feature extractor layer,

# wrap the hub layer in a tf.keras.Sequential model, and add a new classification layer
model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(2)
])
print(model.summary())  # observe model architecture

# compile the model
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# train the model (passing out train and validation data)
history = model.fit(train_batches,
                    epochs=6,
                    validation_data=validation_batches)

# visualize accuracy and loss graphs over epochs (-- optional --)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(6)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# prediction image class
class_names = np.array(info.features['label'].names)
predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]