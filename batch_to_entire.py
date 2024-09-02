import cv2
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--latent', action='store_true', default=False, help='use latent or not')

    parser.add_argument('--step', type=int, default=200, help='step to sample')
    parser.add_argument('--dataset', type=str, default='UCF', help='dataset to eval')

    args = parser.parse_args()

    return args

args = parse_args()
print(args)

if args.latent:
    path = f'results/{args.dataset}/LBBDM-f32/sample_to_eval'
else:
    path = f'results/{args.dataset}/BrownianBridge/sample_to_eval'
sample_num = 0
GT_num = 0
for root,dirs,files in os.walk(path):
    if f'/{args.step}/' in root:
        for file in files:
            if 'sample_from_next' in file:

                save_dir = os.path.join(path,'next')
                    
            elif 'sample_from_prev' in file:
                save_dir = os.path.join(path,'prev')
            try:
                os.mkdir(save_dir)
            except:
                pass
            im = cv2.imread(os.path.join(root,file))
            cv2.imwrite(os.path.join(save_dir,f'{sample_num}.png'),im)
            sample_num += 1
    elif 'ground_truth' in root:
        save_dir = os.path.join(path,'gt')
        try:
            os.mkdir(save_dir)
        except:
            pass
        for file in files:
            im = cv2.imread(os.path.join(root,file))
            cv2.imwrite(os.path.join(save_dir,f'{GT_num}.png'),im)
            GT_num += 1

