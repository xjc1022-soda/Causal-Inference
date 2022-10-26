import torch
import numpy as np
from tqdm import tqdm

def socres_return(dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    img, os, event = dataset[0]
    img = np.expand_dims(img, axis=0)
    for i in tqdm(range(1,len(dataset))):
        #print(i)
        img_new, os_new, event_new = dataset[i]
        img_new = np.expand_dims(img_new, axis=0)
        img = np.concatenate((img, img_new), axis=0)
        os = np.concatenate((os, os_new), axis=0)
        event = np.concatenate((event, event_new), axis=0)
    img = torch.from_numpy(img)
    os = torch.from_numpy(os)
    os = os.to(device)  
    event = torch.from_numpy(event)
    event = event.to(device)
    scores = model(img.float())
    #print(os)
    #print(event)
    #print(scores)
    return img, os, event, scores 


def split_socres(threhold,dataset,model):
    high_data = []
    low_data = []
    high_os = []
    high_event = []
    low_os = []
    low_event = []
    img, os , event, scores = socres_return(dataset,model)
    scores_new = torch.squeeze(scores)
    print("socres return successfully")
    for index, (os_single, event_single, scores_single) in enumerate(tqdm(zip(os,event,scores_new))):
        if scores_single > threhold:
            high_os.append(os_single.item())
            high_event.append(event_single.item())
        else:
            low_os.append(os_single.item())
            low_event.append(event_single.item())
    #print(high_os)
    high_data.append(high_os)
    high_data.append(high_event)
    print(len(high_os))
    low_data.append(low_os)
    low_data.append(low_event)
    print(len(low_os))
    return high_data, low_data


def del_tensor_ele(arr,index):
    if index == len(arr)-1:
        return arr
    else:    
        arr1 = arr[0:index]
        arr2 = arr[index+1:]
        return torch.cat((arr1,arr2),dim=0)
