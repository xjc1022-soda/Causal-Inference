import os
import pydicom
import nibabel as nib
import numpy as np
from tqdm import tqdm
import cv2 as cv
from glob import glob
import random
import ipdb


def read_csv(csv_path):
    properties = {}
    lndex_list = []
    with open(csv_path, "r") as f:
        for line in f.readlines()[1:372]:
            line = line.strip()
            props = line.split(",")
            prop = props[:6] + props[7:22]
            prop = list(map(tran_float, prop))
            idx = int(prop[0])
            event = int(prop[4])
            dict = {"all_prop": prop,
                    "os2": props[3], "event": event, "treatment": props[21]}
            properties[idx] = dict
    print("read csv successfully")
    index_list = list(properties.keys())
    print(index_list)

    return properties, index_list


def tran_float(i):
    if i == "":
        return float(0)
    else:
        return float(i)


def match_img_csv(csv_path, img_path):
    dataset = {}
    properties, csv_index = read_csv(csv_path)
    X, M = load_img(img_path)
    for i in range(np.shape(X)[0]):
        if i in csv_index:
            dataset_i = {"ct": X[i], "mask": M[i],
                         "covariate": properties[i]}
            dataset[idx] = dataset_i
    print("match successfully")
    return dataset, csv_index


def generate_snet_input(dataset, csv_index):
    path = "processed_data"
    os.makedirs(path, exist_ok=True)
    save_npz(path, dataset, csv_index)


def save_npz(path, dataset, index_list):
    print(path)
    for idx in index_list:
        os2 = dataset[idx]["covariate"]["os2"]
        event = dataset[idx]["covariate"]["event"]
        name = idx
        ori_img = dataset[idx]["ct"]
        mask_img = dataset[idx]["mask"]
        npy_name = str(name)+"_"+str(os2)+"_"+str(event)+".npy"
        # print(npy_name)
        ori_dir_name = "single_ori"
        mask_dir_name = "single_mask"
        # print(ori_dir_name)
        os.makedirs(os.path.join(path, ori_dir_name), exist_ok=True)
        os.makedirs(os.path.join(path, mask_dir_name), exist_ok=True)
        ori_path = os.path.join(path, ori_dir_name, npy_name)
        mask_path = os.path.join(path, mask_dir_name, npy_name)
        np.save(ori_path, ori_img)
        np.save(mask_path, mask_img)
    print("save input successfully")




def load_img(img_path):
    whole_img_list = []
    mask_img_list = []
    error_num = 0

    img_files = sorted(glob(os.path.join(img_path, "*/IMG*.dcm")))
    mask_files = sorted(glob(os.path.join(img_path, "*/Untitled.nii")))

    for file in mask_files:
        index_list.append(int(file.split("/")[-2]))

    assert len(img_files) == len(mask_files)

    for whole_path_i, mask_path_i in tqdm(zip(img_files, mask_files), total=len(img_files)):
        assert os.path.exists(whole_path_i), f"{whole_path_i} does not exist!"
        assert os.path.exists(mask_path_i), f"{mask_path_i} does not exist!"

        X = pydicom.dcmread(whole_path_i)
        X = np.array(X.pixel_array)
        m = nib.load(mask_path_i).get_fdata()
        m = np.array(m)
        whole_img_list.append(X)
        mask_img_list.append(m)
        # index_list.append(i)
        # except:
        #     ds = pydicom.dcmread(whole_path_i)
        #     # print(ds.pixel_array)
        #     import ipdb
        #     ipdb.set_trace()
        #     #print("input error:"+str(i))
        #     error_num += 1
    whole_img = np.array(whole_img_list)
    mask_img = np.array(mask_img_list)
    whole_img = np.expand_dims(whole_img_list, axis=-1)
    # print(error_num)
    # print(np.shape(whole_img))
    # print(np.shape(mask_img))
    print("Images loaded successfully")
    print(index_list)
    return whole_img, mask_img


def generate():
    dic_path = "../ct"
    img_path = os.path.join(dic_path, "roi1 540")
    csv_path = os.path.join(dic_path, "covariate_540.csv")
    dataset, csv_index = match_img_csv(csv_path, img_path)
    return dataset, csv_index


dataset, csv_index = generate()
generate_snet_input(dataset, csv_index)
