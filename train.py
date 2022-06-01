from time import sleep
import torch,os
from tqdm import tqdm
import numpy as np
from monai.networks.nets import UNETR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from dataset import CustomDataset
from monai.data import DataLoader,decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete


class Train():
    def __init__(self,args) -> None:
        self.args = args
        self.global_step = 0
        self.dice_val_best = 0.0
        self.global_step_best = 0
        self.epoch_loss_values = []
        self.metric_values = []
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.post_label = AsDiscrete(to_onehot=self.args.n_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.args.n_classes)
        os.makedirs(self.args.save_path,exist_ok=True)
        self.dataset_intialiser()
        self.model_intilaiser()
        self.loss_and_optim_intialiser()
        self.max_iterations = self.args.num_epochs*len(self.train_loader)*self.args.batch_size
        self.eval_num = self.args.val_period
    def model_intilaiser(self):
        self.model = UNETR(in_channels=self.args.in_ch,out_channels=self.args.n_classes,img_size=self.args.train_imz,feature_size=16,\
        hidden_size=768,mlp_dim=3072,num_heads=12,pos_embed="perceptron",norm_name="instance",res_block=True,
        dropout_rate=0.0,).to(device)
    def loss_and_optim_intialiser(self):
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
    def dataset_intialiser(self):
        train_ds = CustomDataset(self.args,is_train=True)
        self.train_loader = DataLoader(train_ds, batch_size = self.args.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=False)
        val_ds = CustomDataset(self.args,is_train=False)
        self.val_loader = DataLoader(val_ds, batch_size = self.args.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=False)
    def validation(self,epoch_iterator_val):
        self.model.eval()
        dice_vals = list()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 80), 4, self.model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    self.post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                self.dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = self.dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (self.global_step, 10.0, dice)
                )
            self.dice_metric.reset()
        mean_dice_val = np.mean(dice_vals)
        return mean_dice_val
    def train(self):
        self.model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].to(device), batch["label"].to(device))
            logit_map = self.model(x)
            loss = self.loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (self.global_step, self.max_iterations, loss)
            )
            if (
                self.global_step % self.eval_num == 0 and self.global_step != 0
            ) or self.global_step == self.max_iterations:
                epoch_iterator_val = tqdm(
                    self.val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = self.validation(epoch_iterator_val)
                epoch_loss /= step
                self.epoch_loss_values.append(epoch_loss)
                self.metric_values.append(dice_val)
                if dice_val > self.dice_val_best:
                    self.dice_val_best = dice_val
                    self.global_step_best = self.global_step
                    torch.save(
                        self.model.state_dict(), os.path.join(self.args.save_path, "best_metric_model.pth")
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            self.dice_val_best, dice_val
                        )
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            self.dice_val_best, dice_val
                        )
                    )
            self.global_step += 1
        return None

        