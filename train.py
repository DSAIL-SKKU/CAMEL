import argparse
import os
import random
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import segmentation_models as sm
os.environ["SM_FRAMEWORK"] = "tf.keras"
from glob import glob
from utils import *
from loss import *
from tf_alloc import allocate

# Set GPU configuration
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

gammas = []

        
# Define a function to print the value of total_loss_weight
def print_total_loss_weight(epoch, logs):
    seg_loss = logs['seg_output_loss']
    clf_loss = logs['cls_output_loss']

    total_weight = seg_loss + clf_loss

    seg_weight = seg_loss / total_weight

    seg_loss_weight.assign(seg_weight)
    gammas.append(seg_weight)
    print("gamma:", seg_loss_weight.numpy())


def ratio_key(target_path, ratio,filename):

    keys = [os.path.splitext(file.split('/')[-1])[0].replace('.npy', '') for file in target_path]
    

    crvo = glob("./2022-OCT-Seg-Data/CRVO/*/*.JPG")
    csc = glob("./2022-OCT-Seg-Data/CSC/*/*.JPG")
    dm = glob("./2022-OCT-Seg-Data/DM/*/*.JPG")
    erm = glob("./2022-OCT-Seg-Data/ERM/*/*.JPG")
    mh = glob("./2022-OCT-Seg-Data/MH/*/*.JPG")
    normal = glob("./2022-OCT-Seg-Data/Normal/*/*.JPG")
    pcv = glob("./2022-OCT-Seg-Data/PCV/*/*.tiff")
    rap = glob("./2022-OCT-Seg-Data/RAP/*/*.tiff")
    wetamd = glob("./2022-OCT-Seg-Data/wetAMD/*/*.tiff")

    crvo = [i.split("/")[-1].split(".JPG")[0] for i in crvo]
    csc = [i.split("/")[-1].split(".JPG")[0] for i in csc]
    dm = [i.split("/")[-1].split(".JPG")[0] for i in dm]
    erm = [i.split("/")[-1].split(".JPG")[0] for i in erm]
    mh = [i.split("/")[-1].split(".JPG")[0] for i in mh]
    normal = [i.split("/")[-1].split(".JPG")[0] for i in normal]
    pcv = [i.split("/")[-1].split(".tif")[0] for i in pcv]
    rap = [i.split("/")[-1].split(".tif")[0] for i in rap]
    wetamd = [i.split("/")[-1].split(".tif")[0] for i in wetamd]

    crvo = [i for i in crvo if i in keys]
    csc = [i for i in csc if i in keys]
    dm = [i for i in dm if i in keys]
    erm = [i for i in erm if i in keys]
    mh = [i for i in mh if i in keys]
    normal = [i for i in normal if i in keys]
    pcv = [i for i in pcv if i in keys]
    rap = [i for i in rap if i in keys]
    wetamd = [i for i in wetamd if i in keys]

    crvo = random.sample(crvo, int(len(crvo) * ratio))
    csc = random.sample(csc, int(len(csc) * ratio))
    dm = random.sample(dm, int(len(dm) * ratio))
    erm = random.sample(erm, int(len(erm) * ratio))
    mh = random.sample(mh, int(len(mh) * ratio))
    normal = random.sample(normal, int(len(normal) * ratio))
    pcv = random.sample(pcv, int(len(pcv) * ratio))
    rap = random.sample(rap, int(len(rap) * ratio))
    wetamd = random.sample(wetamd, int(len(wetamd) * ratio))

    with open(f"./{filename}.csv", "w") as f:
        print("crvo:", len(crvo), file=f)
        print("csc:", len(csc), file=f)
        print("dm:", len(dm), file=f)
        print("erm:", len(erm), file=f)
        print("mh:", len(mh), file=f)
        print("normal:", len(normal), file=f)
        print("pcv:", len(pcv), file=f)
        print("rap:", len(rap), file=f)
        print("wetamd:", len(wetamd), file=f)

    keys = crvo + csc + dm + erm + mh + normal + pcv + rap + wetamd
    # .npy 붙이기 
    keys = [i + ".npy" for i in keys]
    return keys
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", type=str, default="1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save", type=str, default='history')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--aug", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--model", type=str, default='resnet101')
    parser.add_argument("--val_size", type = int, default = 600)
    parser.add_argument("--data_ratio", type = float, default = 0.5)
    parser.add_argument("-img_size",type = int, default = 320)
    parser.add_argument("-erm_weight",type = float, default = 0.01)
    parser.add_argument("-learning_rate",type = float, default = 0.0001)
    parser.add_argument("-weight_decay",type = float, default = 0.0001)
    args = parser.parse_args()

    use_gpu = args.use_gpu
    epochs = args.epochs
    batch_size = args.batch_size
    save_name = args.save
    dropout = args.dropout
    aug = args.aug
    gamma = args.gamma
    model_name = args.model
    val_samples = args.val_size
    ratio = args.data_ratio
    img_size_per = args.img_size
    erm_loss_w = args.erm_weight
    # lr = args.learning_rate
    # wd = args.weight_decay

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


    
    
    ### ================== Dataset Load ================== ###
    input_dir = "./data/annotation_JPG_process/"
    target_dir = './data/annotation_masked_edited_320/'
    input_dir_aug = './data/annotation_JPG_process_edited_320_aug/'
    target_dir_aug = './data/annotation_masked_edited_320_aug/'


    img_size = (320, 320)
    num_classes = 9
    pre_split = False
    

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".JPG")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".npy") and not fname.startswith(".")
        ]
    )

    print("Number of samples:", len(input_img_paths))
    print("Number of masks:", len(target_img_paths))

    # Split our img paths into a training and a validation set
    seed = 42
    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)

    target_img_paths = ratio_key(target_img_paths, ratio,f"image_{aug}_{ratio}_{img_size_per}")
    target_img_paths = [target_dir + i for i in target_img_paths]

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    keys = [os.path.splitext(file.split('/')[-1])[0].replace('.npy', '') for file in target_img_paths]
    print(len(train_input_img_paths), len(train_target_img_paths))
    for key in keys:
        for i in range(0,aug):
            train_input_img_paths.append(input_dir_aug + key + "_" + str(i) +".JPG")
            train_target_img_paths.append(target_dir_aug +  key +"_"+ str(i) +".npy")
    print(len(train_input_img_paths), len(train_target_img_paths))

    matching_paths = []
    for input_path in train_input_img_paths:
        input_filename = os.path.splitext(input_path)[0].split('/')[-1]
        for target_path in train_target_img_paths:
            target_filename = os.path.splitext(target_path)[0].split('/')[-1]
            if input_filename == target_filename:
                matching_paths.append((input_path, target_path))
                break

    print("Number of matching paths: ", len(matching_paths))

    train_input_img_paths, train_target_img_paths = zip(*matching_paths)
    train_input_img_paths = list(train_input_img_paths)
    train_target_img_paths = list(train_target_img_paths)
    
    print("# of Dataset : ",len(train_input_img_paths))
    print("# of Dataset : ",len(val_input_img_paths))

    train_img = convertimg(train_input_img_paths, img_size, True)
    val_img = convertimg(val_input_img_paths, img_size, False)

    print("train_img shape : ", train_img.shape)
    print("val_img shape : ", val_img.shape)
    

    tf.config.run_functions_eagerly(True)

    train_cls_label = []
    val_cls_label = []
    train_seg_label = []
    val_seg_label = []
    train_erm_label = []
    val_erm_label = []
    
    # Seg label
    for path in train_target_img_paths:
        train_seg_label.append(seg_label(path))
    for path in val_target_img_paths:
        val_seg_label.append(seg_label(path))

    # Clf label
    for path in train_input_img_paths:
        train_cls_label.append(extract_label(path))
    for path in val_input_img_paths:
        val_cls_label.append(extract_label(path))
    train_cls_label = tf.keras.utils.to_categorical(train_cls_label, num_classes=9)

    
    # ERM label
    for path in train_input_img_paths:
        train_erm_label.append(extract_erm_label(path))
    for path in val_input_img_paths:
        val_erm_label.append(extract_erm_label(path))
        

    ### ================== Model Compile ================== ###
    
    # 1. Segmentation Model (first branch)
    seg_model = sm.Unet(model_name, classes=9, activation='softmax', input_shape=(320, 320, 3),
                        encoder_weights='imagenet', encoder_freeze=False)
    
    last_layer_name = 'seg_output'
    seg_model.layers[-1]._name = last_layer_name
    seg_model = Model(inputs=seg_model.inputs, outputs=seg_model.outputs, name=seg_model.name)
    
    # 2. Classification Model (second branch)
    clf_model = tf.keras.Model(inputs=seg_model.input, outputs=seg_model.get_layer('relu1').output) # If you use efficientnet, use "get_layer('top_activation')", if use resnet "relu1"
    x = tf.keras.layers.GlobalAveragePooling2D()(clf_model.output)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(9, activation='softmax', name = "cls_output")(x)
    clf_model = tf.keras.Model(inputs=clf_model.input, outputs=x)
    
    # 3. ERM classification Model (third branch)
    erm_model = tf.keras.Model(inputs=seg_model.input, outputs=seg_model.get_layer('relu1').output) # If you use efficientnet, use "get_layer('top_activation')", if use resnet "relu1"
    x = tf.keras.layers.GlobalAveragePooling2D()(erm_model.output)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='erm_output')(x)
    erm_model = tf.keras.Model(inputs=erm_model.input, outputs=x) 
    
    # Define inputs and outputs
    seg_inputs = seg_model.input
    clf_inputs = clf_model.input
    erm_inputs = erm_model.input
    seg_output = seg_model.output
    cls_output = clf_model.output
    erm_output = erm_model.output

    # Define losses for each task
    seg_task_loss = total_loss
    cls_task_loss = 'categorical_crossentropy'
    erm_task_loss = 'binary_crossentropy'

    seg_loss_weight = tf.Variable(initial_value=0.495, trainable=True, name="seg_loss_weight")  #trainable seg_loss 
    erm_loss_weight = erm_loss_w # hyper: 0.2 
    cls_loss_weight = 1.0 - seg_loss_weight - erm_loss_weight

    # Combine losses for each task
    loss = {
        'seg_output': seg_task_loss,
        'cls_output': cls_task_loss,
        'erm_output' : erm_task_loss,
    }

    # Define metrics for each task
    metrics = {
        'seg_output': [dice_coef, dice_coef_multilabel],
        'cls_output': ['accuracy'], 
        'erm_output': ['accuracy'],
    }


    # Compile the model with the combined loss and metrics
    model = tf.keras.Model(inputs=seg_inputs, outputs=[seg_output, cls_output, erm_output])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
                loss=loss, 
                loss_weights=[seg_loss_weight, 0.99-seg_loss_weight, erm_loss_weight], 
                metrics=metrics)

    # schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', min_lr=1e-6)
    
    checkpoint_filepath = f'./final_ckpt/final_{model_name}_{img_size_per}_{aug}_b_{batch_size}_d_{dropout}_ermw_{erm_loss_weight}_r_{ratio}.h5'

    total_loss_weight_callback = LambdaCallback(on_epoch_end=print_total_loss_weight)

    model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            save_weights_only=True,
            mode='auto',
            save_best_only=True)
    
    tb_hist = TensorBoard(log_dir='./test/tb-graph', histogram_freq=0,
                          write_graph=True, write_images=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

    print('Starting training on batch of models for gamma values ', gamma, '\n\n')
    
    model_history = model.fit({'data': np.array(train_img)}, # if you use efficientnet, use "input_1": np.array(train_img), if resnet use "data"
                            {'seg_output' : np.array(train_seg_label), 'cls_output' : np.array(train_cls_label), 'erm_output' : np.array(train_erm_label)}, 
                            validation_split=0.1,
                            batch_size=batch_size, epochs=epochs, 
                            verbose=1,
                            callbacks=[model_checkpoint_callback, tb_hist, early_stopping, total_loss_weight_callback])
                            
     
     
    with open(f'./final_ckpt/mtl/gammas', 'wb') as file_pi:
        pickle.dump(np.array(gammas), file_pi)
        



