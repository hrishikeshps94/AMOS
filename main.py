import argparse
from array import array
import os,tqdm
from train import Train




def main(arg_list):
    if arg_list.mode=='train':
        training_pipe = Train(arg_list)
        training_pipe.train()
    elif arg_list.mode== 'test':
        pass
    else:
        raise Exception('Not selected a valid mode use either train or test')
    return
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Command line arguments for AMOS Grand-challenge')
    main_path = os.path.dirname(os.path.abspath(__file__))
    arg_parser.add_argument('--ds_path',type=str,default = os.path.join('/media/hrishi/data/WORK/RESEARCH/2022/MCCAI-2022/AMOS22','ds'))
    arg_parser.add_argument('--save_path',type=str,default = os.path.join(main_path,'weights'))
    arg_parser.add_argument('--batch_size',type=int,default=1)
    arg_parser.add_argument('--task_type',type=str,default='task1_train.json')
    arg_parser.add_argument('--mode',type=str,default='train')
    arg_parser.add_argument('--in_ch',type=int,default=1)
    arg_parser.add_argument('--n_classes',type=int,default=13)
    arg_parser.add_argument('--train_imz',type=tuple,default=(96, 96, 64))
    arg_parser.add_argument('--val_period',type=int,default=500)
    arg_parser.add_argument('--num_epochs',type=int,default=500)
    arg_list = arg_parser.parse_args()
    main(arg_list)