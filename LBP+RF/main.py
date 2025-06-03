from demo import demo
from model import train_model, valid_model
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODE', default='train',
                        choices=['train', 'valid', 'demo'],
                        help='运行模式: train/valid/demo')
    parser.add_argument('--checkpoint_dir', default='./models',
                        help='模型保存路径')
    parser.add_argument('--train_data', default='./data/fer2013/fer2013.csv',
                        help='训练数据路径')
    parser.add_argument('--valid_data', default='./valid_sets/',
                        help='验证数据目录')
    parser.add_argument('--show_box', action='store_true',
                        help='显示人脸检测框')

    args = parser.parse_args()

    if args.MODE == 'demo':
        demo(args.checkpoint_dir, args.show_box)
    elif args.MODE == 'train':
        train_model(args.train_data, args.checkpoint_dir)
    elif args.MODE == 'valid':
        valid_model(args.checkpoint_dir, args.valid_data)


if __name__ == '__main__':
    main()