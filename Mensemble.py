import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm


def eval_cpu(alphaz, r1, r2, r3, r4, r5, r6, label):
    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]
        _, r55 = r5[i]
        _, r66 = r6[i]
        r = r11 * alphaz[0] + r22 * alphaz[1] + r33 * alphaz[2] + r44 * alphaz[3] + r55 * alphaz[4] + r66 * alphaz[5]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',
                        default=[1],
                        help='weighted summation',
                        type=float)
    parser.add_argument('--dataset',
                        required=True,
                        #default='ntu120/xsub',
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--joint-dir',
                        default='./test_weights/ensemble/NTU120_csub_joint',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        default='./test_weights/ensemble/NTU120_csub_bone',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint2-dir',
                        default='./test_weights/ensemble/NTU120_csub_joint2',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone2-dir',
                        default='./test_weights/ensemble/NTU120_csub_bone2',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint3-dir',
                        default='./test_weights/ensemble/NTU120_csub_joint3',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone3-dir',
                        default='./test_weights/ensemble/NTU120_csub_bone3',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/ntu120/NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('/root/autodl-tmp/NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('/root/autodl-tmp/NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('/root/autodl-tmp/NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint2_dir is not None:
        with open(os.path.join(arg.joint2_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())

    if arg.bone2_dir is not None:
        with open(os.path.join(arg.bone2_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    if arg.joint3_dir is not None:
        with open(os.path.join(arg.joint3_dir, 'epoch1_test_score.pkl'), 'rb') as r5:
            r5 = list(pickle.load(r5).items())

    if arg.bone3_dir is not None:
        with open(os.path.join(arg.bone3_dir, 'epoch1_test_score.pkl'), 'rb') as r6:
            r6 = list(pickle.load(r6).items())

    # 2-Stream acc
    alpha = [0.6, 0.4, 0, 0, 0, 0]
    print('2-Stream acc of MGSAN in', arg.dataset)
    eval_cpu(alpha, r1, r2, r3, r4, r5, r6, label)

    # 4-Stream acc
    alpha = [0.6, 0.4, 0.4, 0.4, 0, 0]
    print('4-Stream acc of MGSAN in', arg.dataset)
    eval_cpu(alpha, r1, r2, r3, r4, r5, r6, label)

    # 6-Stream acc
    alpha = [0.6, 0.4, 0.4, 0.4, 0.4, 0.4]
    print('6-Stream acc of MGSAN in', arg.dataset)
    eval_cpu(alpha, r1, r2, r3, r4, r5, r6, label)
