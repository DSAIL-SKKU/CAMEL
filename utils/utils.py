import pandas as pd
import numpy as np
from PIL import Image
import random
import os 
from glob import glob


# classification label 
def extract_label(path):
    label_df = pd.read_csv("./path_label_total.csv")
    file_name = path.split("/")[-1].split(".")[0]
    
    label = label_df[label_df["path"] == file_name]["label"].values[0]
    if label == "CRVO":
        label = 0
    elif label == "CSC":
        label = 1
    elif label == "DM":
        label = 2
    elif label == "ERM":
        label = 3
    elif label == "MH":
        label = 4
    elif label == "Normal":
        label = 5
    elif label == "PCV":
        label = 6
    elif label == "RAP":
        label = 7
    elif label == "wetAMD":
        label = 8

    return label

def seg_label(path, img_size=(320,320)):
    path = path.replace('tif', 'npy')
    img = np.load(path)
    y = img.astype('int').reshape(img_size + (1,))
    return y


def seg_label_decouple(path, img_size=(320,320)): 
    path = path.replace('tif', 'npy')
    img = np.load(path)

    y = img.astype('int').reshape(img_size + (1,))

    return y



def convertimg(img_paths,img_size,is_training):
    x = []
    for j, path in enumerate(img_paths):

        img = np.array(Image.open(path))
        img = img[:,:,:3]
        img = Image.fromarray(img)     
        img = np.array(img.resize(img_size))
        randn = round(random.uniform(0,1)*100)
        x.append(img)
    return np.array(x)

def class_report(y_pred,y_true,name):
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, accuracy_score

    sensitivty = recall_score(y_true, y_pred, pos_label=1, average='weighted')
    specificity = recall_score(y_true, y_pred, pos_label=0, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    with open(f'./ckpt/multitask/{name}_cr.txt', 'w') as f:
        # print(classification_report(y_true, y_pred), file=f)
        print("accuracy: ", accuracy_score(y_true, y_pred), file=f)
        print("sensitivity: ", sensitivty, file=f)
        print("specificity: ", specificity, file=f)
        print("precision: ", precision, file=f)
        print("f1: ", f1, file=f)
        print("macro_precision: ", macro_precision, file=f)
        print("macro_recall: ", macro_recall, file=f)
        print("macro_f1: ", macro_f1, file=f)
        print("", file=f)
        print("Confusion Matrix: ",file=f)
        print(confusion_matrix(y_true, y_pred), file=f)


# ERM label
def extract_erm_label(path):
    erm_data = pd.read_csv("./erm_label_jh.txt")
    erm_data = list(erm_data['fname'])

            
    erm_data = set(erm_data)
    file_name = path.split("/")[-1].split(".")[0]
    
    if file_name in erm_data:
        return 1
    return 0


def ratio_key(target_path, ratio,filename):

    keys = [os.path.splitext(file.split('/')[-1])[0].replace('.npy', '') for file in target_path]
    

    crvo = glob("./CRVO/*/*.JPG")
    csc = glob("./CSC/*/*.JPG")
    dm = glob("./DM/*/*.JPG")
    erm = glob("./ERM/*/*.JPG")
    mh = glob("./MH/*/*.JPG")
    normal = glob("./Normal/*/*.JPG")
    pcv = glob("./PCV/*/*.tiff")
    rap = glob("./RAP/*/*.tiff")
    wetamd = glob("./wetAMD/*/*.tiff")

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
    keys = [i + ".npy" for i in keys]
    return keys