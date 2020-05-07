
import numpy as np

from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2

# déclaration du nouveau modèle
classifier = Sequential()

# Utilisation du réseau InceptionResNetV2 pré-entrainé sur le jeu de données
# 'imagenet'. Je choisis un format d'entrée d'image 256 * 256 pour plus de
# précisions.
classifier.add(InceptionResNetV2(weights='imagenet',
                                 include_top=False,
                                 input_shape=(256, 256, 3),
                                 pooling='max'))
# Rendre les couches du réseau InceptionResNetV2 non-entrainable (puisque déjà
# entrainés)
for layer in classifier.layers:
    layer.trainable = False

# Ajouter le réseau Dense qui servira à rendre spécifique à la détection Chat vs
# Chien. Le flatten est déjà effectué en fin de InceptionResNetV2
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=1, activation="sigmoid"))


# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(256, 256),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(256, 256),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=5,
                         validation_data=test_set,
                         validation_steps=2000)


test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
