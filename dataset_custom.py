import os
from monai.data import load_decathlon_datalist
from torch.utils.data import Dataset
from monai.transforms import LoadImaged,AddChanneld,Orientationd,Spacingd,ScaleIntensityRanged,CropForegroundd,\
    RandCropByPosNegLabeld,RandFlipd,RandRotate90d,RandShiftIntensityd,ToTensord,Compose,apply_transform
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class CustomDataset(Dataset):
    def __init__(self,args,is_train=True):
        super(Dataset).__init__()
        data_json_path = os.path.join(args.ds_path,args.task_type)
        if is_train:
            # file_datalist = load_decathlon_datalist(data_json_path, True, "test")
            # image_read = LoadImaged(keys=["image", "label"])
            # self.datalist = [image_read(im_name) for im_name in tqdm(file_datalist)]
            self.datalist = load_decathlon_datalist(data_json_path, True, "training")
        else:
            self.datalist = load_decathlon_datalist(data_json_path, True, "test")
        if is_train:
            self.transforms = Compose(
            [LoadImaged(keys=["image", "label"]),AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest"),),
            ScaleIntensityRanged(keys=["image"],a_min=-5798.0,a_max=3284530.8,b_min=0.0,b_max=1.0,clip=True,),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(keys=["image", "label"],label_key="label",spatial_size=(96, 96, 64),pos=1,neg=1,num_samples=4,image_key="image",image_threshold=0,),
            RandFlipd(keys=["image", "label"],spatial_axis=[0],prob=0.10,),
            RandFlipd(keys=["image", "label"],spatial_axis=[1],prob=0.10,),
            RandFlipd(keys=["image", "label"],spatial_axis=[2],prob=0.10,),
            RandRotate90d(keys=["image", "label"],prob=0.10,max_k=3,),
            RandShiftIntensityd(keys=["image"],offsets=0.10,prob=0.50,),
            ToTensord(keys=["image", "label"]),]
            )
        else:
            self.transforms = Compose(
            [LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest"),),
            ScaleIntensityRanged(keys=["image"], a_min=-5798.0, a_max=3284530.8, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),]
            )
    def _transform(self, index: int):
      data_i = self.datalist[index]
      return apply_transform(self.transforms, data_i)

    def __len__(self)-> int:
      return len(self.datalist)

    def __getitem__(self, idx):
      return self._transform(idx)
