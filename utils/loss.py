from tensorflow.keras import backend as K
import tensorflow as tf
from segmentation_models.losses import CategoricalCELoss, CategoricalFocalLoss
import numpy as np 

def dice_coef(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 9 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=9)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])

    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef_multilabel(y_true, y_pred, numLabels=9):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,index], y_pred[:,:,index])
    return dice/numLabels # taking averagehttps://lh3.googleusercontent.com/a/AATXAJwSkTnJqXMgYSOjgcsiJDdOmEpXvWm8wZGUDIUh=s100


def confidence_aware_dice_coef(y_true, y_pred, smooth=1e-7):
    '''
    Confidence-aware Dice coefficient for 9 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement

    weighted dice coefficient
    weight : 1- confidence

    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=9)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    confidences = K.max(y_pred, axis=-1)
    weight = 1 - confidences

    intersect = K.sum(weight * y_true_f * y_pred_f, axis=-1)
    denom = K.sum(weight * y_true_f + y_pred_f, axis=-1)

    return K.mean((2. * intersect / (denom + smooth)))


    

# expected calibration error
# ref :https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d
def expected_calibration_error(confidences, y_pred, y_true, M=10):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get a boolean list of correct/false predictions
    accuracies = y_pred == y_true

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.astype(float).mean()

        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def ECE_loss(y_true, y_pred , M=10):
    # print(y_pred.shape) #(None, 320, 320, 9)
    # print(y_true.shape) #(None, 320, 320, 1)
    
    pred_mask = tf.argmax(y_pred, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    # print(pred_mask.shape) #(None, 320, 320, 1)

    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = tf.nn.softmax(y_pred, axis=-1)
    # print(confidences.shape) # (None, 320, 320, 9)
    
    confidences = tf.reduce_max(confidences, axis=-1)
    # print(confidences.shape) # (None, 320, 320)

    # print(confidences.shape) # (None, 320, 320) np.max(val_preds[idx], axis=2))
    # accuracies = tf.cast(tf.equal(pred_mask, y_true), tf.float32)
    accuracies = tf.cast(tf.equal(tf.cast(pred_mask, tf.float32), tf.cast(y_true, tf.float32)), tf.float32)

    # print(accuracies.shape)

    ece = tf.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = tf.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        prop_in_bin = tf.reduce_mean(tf.cast(in_bin, tf.float32))

        if tf.greater(prop_in_bin, 0):

            accuracy_in_bin = tf.reduce_mean(tf.boolean_mask(accuracies, in_bin))
            avg_confidence_in_bin = tf.reduce_mean(tf.boolean_mask(confidences, in_bin))
            ece += tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    print("ECE Loss:", ece)
    return ece


def total_cal_seg_loss(y_true, y_pred):
    return 0.01*ECE_loss(y_true, y_pred) + 0.99(0.5*(1 - dice_coef(y_true, y_pred)) + 0.5*tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred))


def total_loss(y_true, y_pred):
    return 0.5*(1 - dice_coef(y_true, y_pred)) + 0.5*tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred) 


class MultiLossLayer(tf.keras.layers.Layer):
    def __init__(self, loss_list):
        super(MultiLossLayer, self).__init__()
        self.loss_list = loss_list
        self.sigmas_sq = []
        for i in range(len(self.loss_list)):
            self.sigmas_sq.append(self.add_weight(name='Sigma_sq_' + str(i), shape=[], initializer=tf.initializers.RandomUniform(minval=0.2, maxval=1), trainable=True))

    def call(self, inputs):
        factor = tf.divide(1.0, tf.multiply(2.0, self.sigmas_sq[0]))
        loss = tf.add(tf.multiply(factor, self.loss_list[0]), tf.math.log(self.sigmas_sq[0]))
        for i in range(1, len(self.sigmas_sq)):
            factor = tf.divide(1.0, tf.multiply(2.0, self.sigmas_sq[i]))
            loss = tf.add(loss, tf.add(tf.multiply(factor, self.loss_list[i]), tf.math.log(self.sigmas_sq[i])))
        self.add_loss(loss)
        return inputs


