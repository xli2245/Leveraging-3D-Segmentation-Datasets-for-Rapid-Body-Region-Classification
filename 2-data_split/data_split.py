import os
import json
import argparse
from sklearn.model_selection import KFold

def data_split():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data', help='data path')
    parser.add_argument('--image_name', type=str, default='image', help='image name')
    parser.add_argument('--label_name', type=str, default='label', help='label name')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='train ratio')
    parser.add_argument('--dev_ratio', type=float, default=0.2, help='dev ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')
    parser.add_argument('--n_folds', type=int, default=1, help='Number of folds')
    parser.add_argument('--save_folder', type=str, default='./', help='the folder to save the data split json file')
    args = parser.parse_args()

    data_path = args.data_path
    img_name = args.image_name
    label_name = args.label_name
    train_ratio = args.train_ratio
    dev_ratio = args.dev_ratio
    test_ratio = args.test_ratio
    n_folds = args.n_folds
    save_folder = args.save_folder

    # get the subject list
    subject_list = os.listdir(data_path)
    # exclude the files that are not directories
    subject_list = [subject for subject in subject_list if os.path.isdir(os.path.join(data_path, subject))]
    subject_list.sort()

    if n_folds == 1:
        train_list = subject_list[:int(len(subject_list) * train_ratio)]
        dev_list = subject_list[int(len(subject_list) * train_ratio):int(len(subject_list) * (train_ratio + dev_ratio))]
        test_list = subject_list[int(len(subject_list) * (train_ratio + dev_ratio)):]
        
        # process the lists
        for i in range(len(train_list)):
            train_list[i] = {'image': train_list[i]+'/' + img_name + '.nii.gz', 'label': train_list[i]+'/' + label_name + '.nii.gz'}
        for i in range(len(dev_list)):
            dev_list[i] = {'image': dev_list[i]+'/' + img_name + '.nii.gz', 'label': dev_list[i]+'/' + label_name + '.nii.gz'}
        for i in range(len(test_list)):
            test_list[i] = {'image': test_list[i]+'/' + img_name + '.nii.gz', 'label': test_list[i]+'/' + label_name + '.nii.gz'}
        
        # put them into together
        data = {}
        data['train'] = train_list
        data['dev'] = dev_list
        data['test'] = test_list

        save_path = os.path.join(save_folder, 'data_split.json')
        with open(save_path, 'w') as f:
            json.dump(data, f)
    
    else:
        kf = KFold(n_splits=n_folds, shuffle=True)
        fold_count = 0
        for indices in kf.split(subject_list):
            train_indices = indices[0]
            test_indices = indices[1]
            train_list = [subject_list[i] for i in train_indices]
            test_list = [subject_list[i] for i in test_indices]
            
            dev_list = train_list[:len(train_list) // (n_folds - 1)]
            train_list = train_list[len(train_list) // (n_folds - 1):]
            
            # process the lists
            for i in range(len(train_list)):
                train_list[i] = {'image': train_list[i]+'/' + img_name + '.nii.gz', 'label': train_list[i]+'/' + label_name + '.nii.gz'}
            for i in range(len(dev_list)):
                dev_list[i] = {'image': dev_list[i]+'/' + img_name + '.nii.gz', 'label': dev_list[i]+'/' + label_name + '.nii.gz'}
            for i in range(len(test_list)):
                test_list[i] = {'image': test_list[i]+'/' + img_name + '.nii.gz', 'label': test_list[i]+'/' + label_name + '.nii.gz'}

            fold_data = {}
            fold_data['train'] = train_list
            fold_data['dev'] = dev_list
            fold_data['test'] = test_list
            fold_count += 1

            save_path = os.path.join(save_folder, 'data_split_fold_' + str(fold_count) + '.json')
            with open(save_path, 'w') as f:
                 json.dump(fold_data, f)


if __name__ == '__main__':
    data_split()
