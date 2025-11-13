import argparse
from train import train
from test import test

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train','test'], required=True)
    p.add_argument('--data_root', default='datasets/CAVE')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--model_path', default='checkpoints/best.pth')
    return p.parse_args()

if __name__ == '__main__':
    args = parse()
    if args.mode == 'train':
        train(data_root=args.data_root, epochs=args.epochs, batch_size=args.batch_size)
    else:
        test(data_root=args.data_root, model_path=args.model_path)
