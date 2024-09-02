import cv2
import os
import argparse
from evaluation.FID import calc_FID
from evaluation.LPIPS import calc_LPIPS
from evaluation.FLOLPIPS import calc_FLOLPIPS


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--latent', action='store_true', default=False, help='use latent or not')

    parser.add_argument('--dataset', type=str, default='UCF', help='dataset to eval')
    parser.add_argument('--step', type=int, default=200 , help='step to sample')

    args = parser.parse_args()

    return args

args = parse_args()
print(args)

if args.latent:
    root = f'results/{args.dataset}/LBBDM-f32/sample_to_eval'
else:
    root = f'results/{args.dataset}/BrownianBridge/sample_to_eval'
calc_FID(os.path.join(root,'next'),os.path.join(root,'gt'))
calc_LPIPS(os.path.join(root,f'{args.step}'),os.path.join(root,'ground_truth'))
calc_FLOLPIPS(os.path.join(root,f'{args.step}'),os.path.join(root,'ground_truth'),os.path.join(root,'condition'))
