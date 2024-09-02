import cv2
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--latent', action='store_true', default=False, help='use latent or not')

    parser.add_argument('--dataset', type=str, default='UCF', help='dataset to eval')

    args = parser.parse_args()

    return args

args = parse_args()
print(args)

if args.latent:
    root = f'results/{args.dataset}/LBBDM-f32/sample_to_eval'
else:
    root = f'results/{args.dataset}/BrownianBridge/sample_to_eval'
dirs = ['gt','prev','next']

num_sample = 2048//len(os.listdir(os.path.join(root,'gt'))) ## number of samples needed
print(f'need to copy {num_sample} samples')
for directory in dirs:
    path = os.path.join(root,directory)
    sample_num = 0
    GT_num = 0
    im_list = os.listdir(path)
    count = 0
    for im in im_list:
        img = cv2.imread(os.path.join(path,im))
        for i in range(num_sample):
            cv2.imwrite(os.path.join(path,f'{count}_{i}.png'),img)
        count += 1 


