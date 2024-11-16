################################################
## Image -> Numpy Array 
################################################

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import matplotlib.pyplot as plt


def get_masked_image(image, labels, gradient_threshold=1, color_threshold=1):
    new = np.zeros(image.shape[:2])
    pred = np.zeros(image.shape[:2])
            
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = np.linalg.norm(image[i, j] - labels, axis=1)
            label = np.argmin(temp)
            pred[i, j] = label
            
            if image[i, j][0] == image[i, j][1] == image[i, j][2]: # background
                new[i, j] = 8
            
            else:
                gradient = np.linalg.norm(image[i, j] - image[i-1:i+2, j-1:j+2], axis=1)
                if gradient.max() > gradient_threshold or temp[label] > color_threshold:
                    new[i, j] = label
                else:
                    new[i, j] = 8
                
    mismatch_indices = np.where(new != pred)
    new[mismatch_indices] = pred[mismatch_indices]

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if new[i, j] != 8:
                neighbor_labels = new[i-1:i+2, j-1:j+2]
                unique_labels, label_counts = np.unique(neighbor_labels, return_counts=True)
                label_counts[unique_labels == 8] = 0  
                dominant_label = unique_labels[np.argmax(label_counts)]
                new[i, j] = dominant_label

    new_postprocessed = new.copy()

    return new_postprocessed



# copic,R,G,B,segment
# R32,255,89,0,ERM
# YG23,0,236,255,Retina
# BG02,255,163,162,IRF
# B29,32,128,0,SRF
# E15,204,102,0,SHRM
# YR16,112,0,204,RPE
# E43,255,62,62,PED
# YR00,204,200,0,choroid
# B, 255, 255, 255, Background  


if __name__ == '__main__':
    df = pd.read_csv("2022-OCT-Seg-Data/labels.csv")
    label_idx = []
    labels = ['R32', 'YG23', 'BG02', 'B29', 'E15', 'YR16', 'E43', 'YR00',]
    # R32: ERM
    df = df[df.copic.isin(labels)]

    for row in df.values:
        r, g, b = row[1], row[2], row[3]
        label_idx.append([r,g,b])
    label_idx.append([255, 255, 255]) # background

    # annotation_JPG_process: original 
    samples = os.listdir("./2022-seg/data/annotation_JPG_process/")
    print(len(samples))
    
    # segmentation_layer_edit: new
    ann_path = "2022-OCT-Seg-Data/segmentation_layer_edit/"
    
    erm_label = []

    print("Start")    
    for path in tqdm(samples):
        try:
            path = path.replace('.JPG', '.tiff')
            img = np.array(Image.open(ann_path + path))
            img = img[:,:,:3]
            img = Image.fromarray(img)
            img = np.array(img.resize((480, 480)))
            mask = get_masked_image(img, label_idx)

            # if 0 in mask:
            #     erm_label.append(path)


            idx = path.split("/")[-1].split(" ")[0].replace(".tiff", "")
            np.save(f"./2022-seg/data/annotation_masked_edited_480/{idx}.npy", mask)

            # save img for check
            masked_sample = np.load(f"./2022-seg/data/annotation_masked_edited_480/{idx}.npy")
            pred = np.zeros((masked_sample.shape[0], masked_sample.shape[1], 3))
            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    pred[i, j] = label_idx[int(masked_sample[i, j])]
            
            pred = pred.astype(np.uint8)
            #save img 
            plt.imsave(f"./2022-seg/data_for_check/annotation_masked_edited_480/{idx}.png", pred)
            plt.close()
            
        except Exception as e:
            print(e, path)
    print("Finish")