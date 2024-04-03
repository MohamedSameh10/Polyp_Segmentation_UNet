# Polyp_Segmentation_UNet
## Overview
This repository contains code for polyp segmentation, a critical task in medical image analysis aimed at identifying and delineating polyps in endoscopic images. Polyp segmentation is essential for early detection and treatment of colorectal cancer, improving patient outcomes.

## What is a polyp?
A polyp is a growth on or in an organ in your body. Most polyps are benign, which means they are not cancerous. Some are precancerous, which means that they can turn into cancer over time. Others may be malignant (cancerous), which means they can spread.
Screening tests are important in finding polyps before they become cancerous. These tests also can help find colorectal cancer in its early stages, when you have a good chance of recovery.
The most common screening method is colonoscopy, in which a small tube with a light and camera is inserted into your rectum to look at your colon. If polyps are found, your health care provider may remove them immediately or take tissue samples to send to the lab for analysis.

## Dataset
I utilized CVC-ClinicDB for training and evaluation. The dataset consists of endoscopic images annotated with ground truth polyp masks.

## Results on The Testing Set
- IOU: 0.4826
- Precision: 0.8574
- Recall: 0.6678
- Accuracy: 0.9530

## Model Architecture
```python
def unet_model():
    inputs = Input(shape=(256,256,3))

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    bridge = Conv2D(512,1,activation='relu', padding='same')(pool3)
    bridge = BatchNormalization()(bridge)
    bridge = Conv2D(512,1,activation='relu', padding='same')(bridge)
    bridge = BatchNormalization()(bridge)

    # Decoder
    up1 = Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(bridge)
    merge1 = Concatenate(axis=3)([up1, conv3])
    up1 = Conv2D(256, 3, activation='relu', padding='same')(merge1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(256, 3, activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)

    up2 = Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(up1)
    merge2 = Concatenate(axis=3)([up2, conv2])
    up2 = Conv2D(128, 3, activation='relu', padding='same')(merge2)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(128, 3, activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)

    up3 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(up2)
    merge3 = Concatenate(axis=3)([up3, conv1])
    up3 = Conv2D(64, 3, activation='relu', padding='same')(merge3)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(64, 3, activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(up3)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
