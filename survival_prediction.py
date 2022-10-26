from argparse import ArgumentParser
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18, ResNet18_Weights
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datamodule import SADataModule
from dataset import SADataset
import numpy as np
import wandb 
import km_lifeline as km 
import datetime 
import utils
import time 
from simclr import SimCLR

class SurvivalModel(LightningModule):
    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 layer_num: int,
                 *args,
                 **kwargs) -> None:

        super().__init__()

        self.save_hyperparameters()
        
        self.layer_num = layer_num
        self.img_encoder = resnet18(pretrained=True)
        in_features = self.img_encoder.fc.in_features
        self.img_encoder.fc = nn.Linear(in_features, 1)
        for index, (name,value) in enumerate(self.img_encoder.named_parameters()):
            if index < self.layer_num:
                value.requires_grad= False
            if value.requires_grad == True:
                print("\t", index, name)

    def shared_step(self, batch, batch_idx, split):
        img, os, event = batch
        scores = self.img_encoder(img.float())
        # loss = F.mse_loss(scores, os.float())

        # cox loss
        scores = -scores
        ix = event == 1
        sel_mat = os[ix] <= os
        p_lik = scores[ix] - torch.log(torch.sum(sel_mat.t() * torch.exp(scores).t(), dim=-1))
        loss = -torch.mean(p_lik)
        
        scores = self.img_encoder(img.float())
        ix_ci = torch.where((os < os.squeeze()) & event.bool())
        s1 = scores[ix_ci[0]]
        s2 = scores[ix_ci[1]]
        ci = torch.mean(torch.lt(s1,s2).type(torch.FloatTensor))
        
        if split == 'train':            
            self.log("train_ci", ci, prog_bar=True, on_epoch=True)
            wandb.log({"train_ci": ci})
        elif split == 'val':
            wandb.log({"val_ci": ci})
        # if split == "train":
        #     import matplotlib.pyplot as plt
        #     fig, axes = plt.subplots(3, 3)
        #     for i in range(3):
        #         for j in range(3):
        #             idx = 3 * i + j
        #             axes[i][j].imshow(img[idx, 0].cpu().numpy()*0.5 + 0.5)
        #     plt.savefig("train.png")
        # elif split == "val":
        #     import matplotlib.pyplot as plt
        #     fig, axes = plt.subplots(3, 3)
        #     for i in range(3):
        #         for j in range(3):
        #             idx = 3 * i + j
        #             axes[i][j].imshow(img[idx, 0].cpu().numpy()*0.5 + 0.5)
        #     plt.savefig("val.png")

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        #self.log("train_ci", ci, prog_bar=True, on_step=True)
        wandb.log({"train loss": loss})
        #wandb.log({"train ci": ci})

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "val")
        self.log("val_loss", loss, prog_bar=True)
        #self.log("val_ci", ci, prog_bar=True)
        wandb.log({"val_loss": loss})
        #wandb.log({"validation ci": ci})
        return loss

    '''
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "test")
        self.log("test_loss", loss, prog_bar=True)

        return loss
    '''

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()),  
                          lr=self.hparams.learning_rate
                          )
        lr_scheduler = StepLR(optimizer, step_size=20, gamma=1.)
        # lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            # "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--num_split", type=int, default=5)
        parser.add_argument("--layer_num", type=int, default=58)
        return parser


def cli_main():
    seed_everything(42)



    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--valid_pos", type=int, default=0)
    parser.add_argument("--dataset_model", type=str, default="sim")
    parser.add_argument("--km_plot",type=bool, default=True)
    parser.add_argument("--val_ci", type=bool, default=True)

    if args.dataset_model == "sim":
        parser = SimCLR.add_model_specific_args(parser)
        args.deterministic = True
        args.log_every_n_steps = 25
        args.accelerator = 'gpu'
        args.devices = [0]  
    else:    
        parser = SurvivalModel.add_model_specific_args(parser)
        args = parser.parse_args()
        args.max_epochs = 100
        args.deterministic = True
        args.log_every_n_steps = 25
        args.accelerator = 'gpu'
        args.devices = [0]    
    


    wandb.init(project="causal project"+str(datetime.date.today()),
               name=str(args.layer_num)+": "+str(args.learning_rate)+"_"+str(args.max_epochs)
              )    

    if args.dataset_model == "sim":
        model = SimCLR(**args.__dict__)
    else:
        model = SurvivalModel(**args.__dict__)
      

    dm = SADataModule(args.batch_size, args.num_workers, args.num_split, args.valid_pos, args.dataset_model)

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(dirpath="ckpts", monitor="val_loss", save_last=True, save_top_k=1)
        ])
    trainer.fit(model, dm)
    

    if args.km_plot:
        dataset_train = SADataset(args.num_split, args.valid_pos, args.dataset_model, split="train")
        high_train, low_train = utils.split_socres(-0.04,dataset_train,model.img_encoder)
        km.plot_kmf(high_train,low_train, "train")

        val_dataset = SADataset(args.num_split, args.valid_pos, args.dataset_model, split="val")
        high_val, low_val = utils.split_socres(-0.04,val_dataset,model.img_encoder)
        km.plot_kmf(high_val,low_val, "val") 


    if args.val_ci:
        print("lenth of val_dataset "+str(len(val_dataset)))
        img, os, event, scores = utils.socres_return(val_dataset, model.img_encoder)
        os = torch.unsqueeze(os, dim=1)
        ix= torch.where(os < os.squeeze())
        indices = torch.where(event.bool())
        num_delete = 0
        for i in range(len(ix[0])):
            if ix[0][i-num_delete].item() in indices[0]:
                pass
            else:
                ix =(utils.del_tensor_ele(ix[0], i-num_delete), utils.del_tensor_ele(ix[1], i-num_delete))
                num_delete += 1

        s1 = scores[ix[0]]
        s2 = scores[ix[1]]
        ci = torch.mean(torch.lt(s1,s2).type(torch.FloatTensor))
        print("Average c index is "+str(ci))


if __name__ == "__main__":
    cli_main()
