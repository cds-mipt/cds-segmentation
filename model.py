from keras.models import *
from keras.layers import *
from keras.optimizers import *


smooth = 1.

#special metrics for FCN training on small blobs - Dice
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_sparsed(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    Npixels = K.sum(y_true_f * y_true_f)
    dice_common = (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

    dice_coef_sparsed_value = K.switch(Npixels < 10, 0.0, dice_common)
    return dice_coef_sparsed_value

def dice_coef_sparsed_1(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    Npixels = K.sum(y_true_f * y_true_f)
    Npixels_pred = K.sum(y_pred_f * y_pred_f)
    dice_common = (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

    dice_coef_sparsed_value = K.switch(Npixels < 10, K.switch(Npixels_pred < 10, 0.0, dice_common), dice_common) #to check
    return dice_coef_sparsed_value

#loss metrics for FCN training on base of Dice
def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def dice_coef_multilabel(y_true, y_pred, numLabels=10, coefs = [2, 3, 10, 3, 1, 3, 1, 1, 1, 1]):
    dice=0
    for index in range(numLabels):
        dice += coefs[index]*dice_coef_sparsed(y_true[:,:,:,index], y_pred[:,:,:,index]) #output tensor have shape (batch_size,
                                                                    # width, height, numLabels)
    return dice/numLabels

def dice_coef_multilabel_1(y_true, y_pred, numLabels=10, coefs = [2, 3, 10, 3, 1, 2, 1, 1, 1, 1]):
    dice=0
    for index in range(numLabels):
        dice += coefs[index]*dice_coef_sparsed_1(y_true[:,:,:,index], y_pred[:,:,:,index]) #output tensor have shape (batch_size,
                                                                    # width, height, numLabels)
    return dice/numLabels

def dice_coef_multilabel_loss(y_true, y_pred):
    return 1.-dice_coef_multilabel(y_true, y_pred)

def dice_coef_n_true(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f)
    Npixels = K.sum(y_true_f * y_true_f)
    return Npixels

def dice_coef_n_inter(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    #Npixels = K.sum(y_true_f * y_true_f)
    return intersection

def dice_coef_n_pred(y_true, y_pred):
    #y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f)
    Npixels = K.sum(y_pred_f * y_pred_f)
    return Npixels

def dice_0(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,0], y_pred[:,:,:,0]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_1(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,1], y_pred[:,:,:,1]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_2(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,2], y_pred[:,:,:,2]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_3(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,3], y_pred[:,:,:,3]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_4(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,4], y_pred[:,:,:,4]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_5(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,5], y_pred[:,:,:,5]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_6(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,6], y_pred[:,:,:,6]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_7(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,7], y_pred[:,:,:,7]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_8(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,8], y_pred[:,:,:,8]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)

def dice_9(y_true, y_pred):
    return dice_coef_sparsed(y_true[:,:,:,9], y_pred[:,:,:,9]) #output tensor have shape (batch_size, width, height, numLabels)                                                             # width, height, numLabels)





# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T

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


