# Image Classification Task

# 1. Import all necessary libraries.

import os
import glob

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator

num_classes = 2  # Change according to your application
batch_size = 32

if num_classes == 2:
    last_layer_classifier = num_classes - 1
    cl_mode = 'binary'
    classifier_loss = 'binary_crossentropy'
else:
    last_layer_classifier = num_classes - 1
    cl_mode = 'categorical'
    classifier_loss = 'categorical_crossentropy'

# 2. Data pre-processing

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

main_dir = os.getcwd()
output_dir = main_output_dir + 'training_set/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('Download training data abd place it in training_set directory'

    output_dir = main_output_dir + 'test_set/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Download test data abd place it in test_set directory'

    train_samples = 8000  # change according to your application
    test_samples = 2000  # change according to your application

    training_set = train_datagen.flow_from_directory('training_set', target_size=(224, 224), batch_size=batch_size,
                                                     class_mode=cl_mode)
    test_set = test_datagen.flow_from_directory('test_set', target_size=(224, 224), batch_size=batch_size,
                                                class_mode=cl_mode)

    # 3. import pre-trained network

    model = VGG16(weights='imagenet', include_top=True)
    print(model.summary())

    # 4. Remove last layer of the pre-trained network and add layer according to available classes

    x = Dense(last_layer_classifier, activation='softmax', name='predictions')(model.layers[-2].output)
    my_model = Model(input=model.input, output=x)
    print(my_model.summary())

    # 5. List of callbacks

    weight_name = 'image_classification_model_weights.h5'
    checkpoint = ModelCheckpoint(weight_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', cooldown=0,
                           min_lr=0.000000001)
    callbacks_list = [checkpoint, lr, early]

    # 6. Model compile and train the model

    adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    my_model.compile(optimizer=adam, loss=classifier_loss, metrics=['accuracy'])
    history = my_model.fit_generator(training_set, steps_per_epoch=round(len(train_samples) / batch_size, epochs=25,
                                                                         validation_data=test_set,
                                                                         validation_steps=round(
                                                                             len(test_samples) / batch_size,
                                                                             callbacks=callbacks_list, shuffle=True,
                                                                             verbose=1)

    # 7. Plot the curves

    # Accuracy curve
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model categorical accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.close()

    # Loss curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss.png')
