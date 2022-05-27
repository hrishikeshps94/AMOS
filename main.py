import argparse
from array import array
import os,tqdm
from dataset import CustomDataset
from monai.data import DataLoader



def main(arg_list):
    if arg_list.mode=='train':
        train_ds = CustomDataset(arg_list,is_train=True)
        # train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=os.cpu_count(), pin_memory=False)
        for i in tqdm.tqdm(train_ds):
            pass

        # training_pipe = Train(arg_list)
        # training_pipe.run()
    elif arg_list.mode== 'test':
        pass
        # testing_pipe = Test(arg_list)
        # testing_pipe.run()
    else:
        raise Exception('Not selected a valid mode use either train or test')
    return
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Command line arguments for AMOS Grand-challenge')
    main_path = os.path.dirname(os.path.abspath(__file__))
    arg_parser.add_argument('--ds_path',type=str,default = '/media/hrishi/data/WORK/RESEARCH/2022/MCCAI-2022/AMOS22/ds')
    arg_parser.add_argument('--batch_size',type=int,default=1)
    arg_parser.add_argument('--task_type',type=str,default='task1_train.json')
    arg_parser.add_argument('--mode',type=str,default='train')
    arg_list = arg_parser.parse_args()
    main(arg_list)