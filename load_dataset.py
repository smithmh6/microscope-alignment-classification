"""
Load the dataset and save as compressed .npz file.
"""

# standard imports
import os
from pathlib import Path
import threading
import queue

# external dependencies
import cv2 as cv
import numpy as np
from tqdm import tqdm
import tensorflow as tf

def read_img_task(img_dir, q):
    """ Read images and return lists """

    print(f"Folder {img_dir} started.")

    img_list = []
    for f in os.listdir(img_dir):
        img_path = os.path.join(img_dir, f)
        if img_path.endswith('.tif'):
            img_list.append(img_path)

    for i, p in enumerate(img_list):

        # check the classification
        if "noalign" in str(p):
            label = tf.keras.utils.to_categorical(0, 2)
        else:
            label = tf.keras.utils.to_categorical(1, 2)

        # read the image data
        img = cv.imread(str(p), -1)

        # update the images and labels
        q.put((img, label))

        #print(label, img.shape, type(img), img.dtype, np.max(img), np.min(img)

    print(f"Folder {img_dir} Complete.")

def compress_data(ds_path:str, out_path:str):
    """ Read images and compress dataset to .npz file """

    threads = list()
    q = queue.Queue()

    for folder in os.listdir(ds_path):
        for sub in os.listdir(os.path.join(ds_path, folder)):
            t = threading.Thread(
                target=read_img_task,
                args=(os.path.join(ds_path, folder, sub), q))
            threads.append(t)
            t.start()

    print("joining threads")
    # join all threads in list
    for thread in threads:
        thread.join()

    # extract all items from queue
    images = list()
    labels = list()

    print("Processing image data")
    while not q.empty():
        data = q.get()
        images.append(data[0])
        labels.append(data[1])

    print(f"Saving file {out_path}.npz")

    # save compressed .npz when finished
    np.savez_compressed(
        Path(out_path),
        images=images,
        labels=labels,
        classes=['not-aligned', 'aligned'])

def main():
    """ main script """

    out_dir = r'C:\Users\hsmith\datasets\tsi_image_recognition'

    train_data = Path(r'P:\\TSW\\hsmith\\Calibration data set\TrainingData')
    x = threading.Thread(target=compress_data, args=(train_data, os.path.join(out_dir, 'train')))
    x.start()

    val_data = Path(r'P:\\TSW\\hsmith\\Calibration data set\ValidationData')
    y = threading.Thread(target=compress_data, args=(val_data, os.path.join(out_dir, 'val')))
    y.start()

    test_data = Path(r'P:\\TSW\\hsmith\\Calibration data set\TestingData')
    z = threading.Thread(target=compress_data, args=(test_data, os.path.join(out_dir, 'test')))
    z.start()

    for thread in [x, y, z]:
        thread.join()

    print("Completed!")

if __name__ == "__main__":
    main()
