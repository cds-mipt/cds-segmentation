from keras.models import *
from keras.layers import *
from keras.optimizers import *


smooth = 1.

#special metrics for FCN training on small blobs - Dice
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_sparsed(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    Npixels = K.sum(y_true_f * y_true_f)
    dice_common = (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

    dice_coef_sparsed_value = K.switch(Npixels < 10, 0.0, dice_common)
    return dice_coef_sparsed_value

#loss metrics for FCN training on base of Dice
def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def dice_coef_multilabel(y_true, y_pred, numLabels=13):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]) #output tensor have shape (batch_size,
                                                                    # width, height, numLabels)
    return dice/numLabels

def dice_coef_multilabel_1(y_true, y_pred, numLabels=10, coefs = [2, 3, 10, 3, 1, 2, 1, 1, 1, 1]):
    dice = 0
    for index in range(numLabels):
        dice += coefs[index] * dice_coef_sparsed(y_true[:, :, :, index],
                                                 y_pred[:, :, :, index])  # output tensor have shape (batch_size,
    return dice / numLabels

def dice_coef_multilabel_loss(y_true, y_pred):
    return 1.-dice_coef_multilabel(y_true, y_pred)


def dice_0(y_true, y_pred):
    return dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0])
def dice_1(y_true, y_pred):
    return dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1])
def dice_2(y_true, y_pred):
    return dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2])
def dice_3(y_true, y_pred):
    return dice_coef(y_true[:,:,:,3], y_pred[:,:,:,3])
def dice_4(y_true, y_pred):
    return dice_coef(y_true[:,:,:,4], y_pred[:,:,:,4])
def dice_5(y_true, y_pred):
    return dice_coef(y_true[:,:,:,5], y_pred[:,:,:,5])
def dice_6(y_true, y_pred):
    return dice_coef(y_true[:,:,:,6], y_pred[:,:,:,6])
def dice_7(y_true, y_pred):
    return dice_coef(y_true[:,:,:,7], y_pred[:,:,:,7])
def dice_8(y_true, y_pred):
    return dice_coef(y_true[:,:,:,8], y_pred[:,:,:,8])
def dice_9(y_true, y_pred):
    return dice_coef(y_true[:,:,:,9], y_pred[:,:,:,9])
def dice_10(y_true, y_pred):
    return dice_coef(y_true[:,:,:,10], y_pred[:,:,:,10])
def dice_11(y_true, y_pred):
    return dice_coef(y_true[:,:,:,11], y_pred[:,:,:,11])
def dice_12(y_true, y_pred):
    return dice_coef(y_true[:,:,:,12], y_pred[:,:,:,12])
def dice_13(y_true, y_pred):
    return dice_coef(y_true[:,:,:,13], y_pred[:,:,:,13])


def unet_light(pretrained_weights = None,input_size = (256,256,1),learning_rate = 1e-4, n_classes = 1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop3))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(n_classes, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    if n_classes == 1:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    else:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def unet_light_ct(pretrained_weights = None,input_size = (256,256,1),learning_rate = 1e-4, n_classes = 1, no_compile = False):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(drop3)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    #up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(n_classes, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    if no_compile == False:
        if n_classes == 1:
            model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
        else:
            model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def unet_light_ct_tv(pretrained_weights = None,input_size = (256,256,1),learning_rate = 1e-4, n_classes = 1, no_compile = False):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    #up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop3))
    #Jason Brownlee. How to use the UpSampling2D and Conv2DTranspose Layers in Keras. 2019 https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
    #TensorRT Support Matrix. TensorRT 5.1.5. https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-support-matrix/index.html
    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(drop3)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    #up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(n_classes, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    if no_compile == False:
        if n_classes == 1:
            model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
        else:
            model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_multilabel_loss,
                          metrics=[dice_coef_multilabel,
                                   dice_0,dice_1,dice_2,dice_3,dice_4,dice_5,dice_6,dice_7,dice_8,dice_9])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def unet_light_mct(pretrained_weights = None,input_size = (256,256,1),learning_rate = 1e-4, n_classes = 1, no_compile = False):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 5, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv11 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv11)
    conv2 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool1)
    conv21 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv2)
    conv22 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)
    conv3 = Conv2D(128, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    # up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop3))
    # Jason Brownlee. How to use the UpSampling2D and Conv2DTranspose Layers in Keras. 2019 https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
    # TensorRT Support Matrix. TensorRT 5.1.5. https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-support-matrix/index.html
    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(drop3)
    merge8 = concatenate([conv22, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8)
    merge9 = concatenate([conv11, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(n_classes, 1, activation='softmax')(conv9)

    model = Model(input=inputs, output=conv10)

    if no_compile == False:
        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',
                      metrics=[dice_coef_multilabel,
                               dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, dice_7, dice_8, dice_9,
                               dice_10, dice_11, dice_12])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
