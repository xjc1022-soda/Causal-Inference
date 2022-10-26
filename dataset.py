import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import ipdb
import torchio as tio
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform


class SADataset(Dataset):
    def __init__(self, num_split, valid_pos, dataset_model, data_dir="processed_data", split="train") -> None:
        super().__init__()

        img_ori_dir = os.path.join(data_dir, "single_ori")
        mask_dir = os.path.join(data_dir, "single_mask")

        self.datasets = []
        self.datasets_name = []
        f_list = os.listdir(img_ori_dir)


        def get_k_folder(f_list, num_split, valid_pos):
            assert num_split > 1
            folder_size = len(f_list) // num_split
            #print(folder_size)
            train_list = None
            for j in range(num_split):
                idx = slice(j * folder_size, (j + 1) * folder_size)
                list_part = f_list[idx]
                if j == valid_pos:
                    valid_list = list_part
                elif train_list == None:
                    train_list = list_part
                else:
                    train_list += list_part
            return train_list, valid_list

        def data_augment(X, transform_order):
            elastic_transform = tio.RandomElasticDeformation(
                num_control_points=5,
                locked_borders=1,
            )

            affine_transform = tio.RandomAffine(
                scales=(1, 1.2),
                degrees=30,
            )

            blur_transform = tio.RandomBlur(std=0.5)

            anisotropy_transform = tio.RandomAnisotropy(
                axes=(0, 1),
                downsampling=(4, 6),
            )
            
            noise_transform = tio.RandomNoise(
                    mean=0.5, 
                    std=(0,0.25),
            )

            if transform_order == "blur":
                aug_tran = blur_transform
            elif transform_order == "affine":
                aug_tran = affine_transform
            elif transform_order == "anisotropy":
                aug_tran = anisotropy_transform
            elif transform_order == "elastic":
                aug_tran = elastic_transform
            elif transform_order == "noise":
                aug_tran = noise_transform

            X_new = aug_tran(np.expand_dims(X, axis=0))
            X_new = np.squeeze(X_new, axis=0)
            X_new = (X_new - np.min(X_new[X_new > 0])) / (np.max(X_new[X_new > 0]) - np.min(X_new[X_new > 0]))


            return X_new


        def generate_folder_dataset(file_list, split):
            self.datasets = []
            self.datasets_name = []
            #print(file_list)
            #print(len(file_list))
            for index, i in enumerate(file_list):
                #print(os.path.splitext(i)[0])
                name, os2, event = os.path.splitext(i)[0].split('_')
                X = np.load(os.path.join(img_ori_dir, i))
                m = np.load(os.path.join(mask_dir, i))
                m = np.squeeze(m)
                X0 = X.copy()
                m0 = m.copy()

                X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                X = clahe.apply(X)
                X = (X - np.min(X)) / (np.max(X) - np.min(X))

                x, y = np.where(m > 0)
                w0, h0 = m.shape
                x_min = max(0, int(np.min(x) - 5))
                x_max = min(w0, int(np.max(x) + 5))
                y_min = max(0, int(np.min(y) - 5))
                y_max = min(h0, int(np.max(y) + 5))

                # print (x_min, x_max, y_min, y_max)
                m = m[x_min:x_max, y_min:y_max]
                X = X[x_min:x_max, y_min:y_max]

                X_m_1 = X.copy()
                X_m_1 = (X_m_1 - np.min(X_m_1[m > 0])) / (np.max(X_m_1[m > 0]) - np.min(X_m_1[m > 0]))
                X_m_1[m == 0] = 0


                h, w = X_m_1.shape

                if h < w:
                    pad_1 = (w - h) // 2
                    pad_2 = w - pad_1 - h
                    X_m_1 = np.lib.pad(X_m_1, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0, 0))
                    m = np.lib.pad(m, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0, 0))
                elif h >= w:
                    pad_1 = (h - w) // 2
                    pad_2 = h - pad_1 - w
                    X_m_1 = np.lib.pad(X_m_1, ((0, 0), (pad_1, pad_2)), 'constant', constant_values=(0, 0))
                    m = np.lib.pad(m, ((0, 0), (pad_1, pad_2)), 'constant', constant_values=(0, 0))

                if X_m_1.shape[0] != 160 or X_m_1.shape[1] != 160:
                    X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_CUBIC)
                    m = cv2.resize(m, (160, 160), interpolation=cv2.INTER_CUBIC)

                if m0.shape[0] != 160 or m0.shape[1] != 160:
                    m0 = cv2.resize(m0, (160, 160), interpolation=cv2.INTER_CUBIC)

                X_m_1 = (X_m_1 - np.min(X_m_1[m > 0])) / (np.max(X_m_1[m > 0]) - np.min(X_m_1[m > 0]))
                X_m_1[m <= 0] = 0
                X_m_1 = np.expand_dims(X_m_1, axis=2)



                if dataset_model == "aug" and split == "train":
                    '''
                    X_m_2 = data_augment(X_m_1, "blur")
                    XX_2 = np.concatenate((X_m_2, X_m_2, X_m_2), axis=-1)
                    self.datasets.append((XX_2, np.array([float(os2)]), np.array([int(event)])))
                    self.datasets_name.append(name)
                    
                    X_m_3 = data_augment(X_m_1, "affine")
                    XX_3 = np.concatenate((X_m_3, X_m_3, X_m_3), axis=-1)
                    self.datasets.append((XX_3, np.array([float(os2)]), np.array([int(event)])))
                    self.datasets_name.append(name)
                    
                    X_m_4 = data_augment(X_m_1, "elastic")
                    XX_4 = np.concatenate((X_m_4, X_m_4, X_m_4), axis=-1)
                    self.datasets.append((XX_4, np.array([float(os2)]), np.array([int(event)])))
                    self.datasets_name.append(name)
                    '''
                    X_m_5 = data_augment(X_m_1, "anisotropy")
                    XX_5 = np.concatenate((X_m_5, X_m_5, X_m_5), axis=-1)
                    self.datasets.append((XX_5, np.array([float(os2)]), np.array([int(event)])))
                    self.datasets_name.append(name)

                    X_m_6 = data_augment(X_m_1, "noise")
                    XX_6 = np.concatenate((X_m_6, X_m_6, X_m_6), axis=-1)
                    self.datasets.append((XX_6, np.array([float(os2)]), np.array([int(event)])))
                    self.datasets_name.append(name)
                    


                m = np.expand_dims(m, axis=2)
                m0 = np.expand_dims(m0, axis=2)

                if dataset_model=="sim" and split=="trian":
                    simclr_transform = SimCLRTrainDataTransform(input_height=160)
                    X_sim = simclr_transform(np.expand_dims(X_m_1, axis=0))
                    X_sim = np.squeeze(X_sim, axis=0)
                    X_sim = (X_sim - np.min(X_sim[X_sim > 0])) / (np.max(X_sim[X_sim > 0]) - np.min(X_sim[X_sim > 0]))
                    XX_sim = np.concatenate((X_sim, X_sim, X_sim), axis=-1)
                    self.datasets.append((XX_sim, np.array([float(os2)]), np.array([int(event)])))
                    self.datasets_name.append(name)                   
                elif dataset_model=="sim" and split=="val":
                    simclr_transform = SimCLRValDataTransform(input_height=160)
                    X_sim = simclr_transform(np.expand_dims(X_m_1, axis=0))
                    X_sim = np.squeeze(X_sim, axis=0)
                    X_sim = (X_sim - np.min(X_sim[X_sim > 0])) / (np.max(X_sim[X_sim > 0]) - np.min(X_sim[X_sim > 0]))
                    XX_sim = np.concatenate((X_sim, X_sim, X_sim), axis=-1)
                    self.datasets.append((XX_sim, np.array([float(os2)]), np.array([int(event)])))
                    self.datasets_name.append(name) 

                XX = np.concatenate((X_m_1, X_m_1, X_m_1), axis=-1)

                self.datasets.append((XX, np.array([float(os2)]), np.array([int(event)])))
                self.datasets_name.append(name)
                '''
                import matplotlib.pyplot as plt
                plt.imshow(X_m_1)
                plt.colorbar()
                plt.savefig("test_ori.png")
                plt.imshow(X_m_6)
                plt.savefig("test_6.png")
                plt.imshow(X_m_5)
                plt.savefig("test_5.png")
                ipdb.set_trace()
                '''
            return self.datasets, self.datasets_name



        train_list, valid_list = get_k_folder(f_list, num_split, valid_pos)
        #print("valid_pos is "+str(valid_pos))

        if split == "train":
            self.datasets, self.datasets_name = generate_folder_dataset(train_list, "train")
        else:
            self.datasets, self.datasets_name = generate_folder_dataset(valid_list, "val")


        if split == "train":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop(size=(160, 160)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        img, os, event = self.datasets[index]
        #img = self.transform(img)
    
        return img, os, event


if __name__ == "__main__":
    dataset = SADataset()
    print(dataset[0])
    print(len(dataset))
