from src.test import test_model
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('-l','--load_model_path', type=str, default='ckpt/model.toy', help='load model from path')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size ')
    parser.add_argument('--show_stat', type=int, default=1, help='Show stat at every batch')
    parser.add_argument('-n','--num_epochs', type=int, default=2000, help='See variable name')
    parser.add_argument('-d','--depth', type=int, default=0, help='See variable name')
    parser.add_argument('-lr','--lr', type=float, default=1e-5, help='See variable name')
    parser.add_argument('--step_size', type=int, default=100, help='See variable name')
    parser.add_argument('--gamma', type=float, default=0.8, help='See variable name')
    parser.add_argument('--lambda_n', type=float, default=1e-4, help='See variable name')
    parser.add_argument('--lambda_lap', type=float, default=0.6, help='See variable name')
    parser.add_argument('--lambda_e', type=float, default=0.198, help='See variable name')
    parser.add_argument('--data_dir', type=str, default='data/1/', help='See variable name')
    parser.add_argument('--suffix', type=str, default='test', help='See variable name')
    parser.add_argument('--feature_scale', type=int, default=10, help='See variable name')
    parser.add_argument('--dim_size', type=int, default=2, help='See variable name')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parseArgs()
    print(args)
    test_model(args)