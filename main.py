from src.train_rl import train_model
# from src.test import test_model
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('-l','--load_model_dirpath', type=str, default='', help='load model from path')
    parser.add_argument('-s','--save_model_dirpath', type=str, default='ckpt/', help='save model to path')
    parser.add_argument('-bs','--batch_size', type=int, default=100, help='Batch size ')
    parser.add_argument('--show_stat', type=int, default=1, help='Show stat at every batch')
    parser.add_argument('-sf','--sf', type=str, default='', help='suffix_name for pred')
    parser.add_argument('-n','--num_epochs', type=int, default=2000, help='See variable name')
    parser.add_argument('-d','--depth', type=int, default=0, help='See variable name')
    parser.add_argument('-lr','--lr', type=float, default=1e-5, help='See variable name')
    parser.add_argument('--step_size', type=int, default=100, help='See variable name')
    parser.add_argument('--gamma', type=float, default=0.8, help='See variable name')
    parser.add_argument('--lambda_n', type=float, default=1e-5, help='See variable name')
    parser.add_argument('--lambda_lap', type=float, default=0.2, help='See variable name')
    parser.add_argument('--lambda_e', type=float, default=0.1, help='See variable name')
    parser.add_argument('--data_dir', type=str, default='data/1/', help='See variable name')
    parser.add_argument('--suffix', type=str, default='train', help='See variable name')
    parser.add_argument('--feature_scale', type=int, default=10, help='See variable name')
    parser.add_argument('--dim_size', type=int, default=2, help='See variable name')
    parser.add_argument('--img_width', type=int, default=600, help='See variable name')
    parser.add_argument('--img_height', type=int, default=600, help='See variable name')
    parser.add_argument('-t','--test', dest='test', default = False, action='store_true',help='See variable name')
    parser.add_argument('--add_prob', type=float, default=0.5, help='See variable name')
    parser.add_argument('--num_polygons', type=int, default=3, help='See variable name')
    parser.add_argument('-i','--iters_per_block', type=int, default=100, help='See variable name')
    parser.add_argument('--load_rl_count', type=int, default=200, help='See variable name')
    args = parser.parse_args()

    return args

if __name__=="__main__":
    args = parseArgs()
    print(args)
    if(args.test):
        # test_model(args)
        pass
    else:
        train_model(args)