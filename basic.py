import os
import argparse
import glob
import logging
import librosa
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle


# Argument Parsing
parser = argparse.ArgumentParser(description="Model Training Script")
parser.add_argument('--checkpoints_path', type=str, help='Checkpoints Path')
parser.add_argument('--result_path', type=str, help='Result Path')
parser.add_argument('--log_path', type=str, help='Log Path')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight Decay')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of Epochs')
parser.add_argument('--gpu_id', type=int, default=3, help='GPU ID')
parser.add_argument('--num_test', type=int, default=0, help='Number of Tests')
parser.add_argument('--feature_type', type=str, default='stft', help='Feature Type')
parser.add_argument('--feature_time', type=float, default=0.5, help='Feature Time Size')
parser.add_argument('--n_fft', type=int, default=4096, help='Number of FFT components')
parser.add_argument('--win_length', type=int, default=4096, help='Window Length')
parser.add_argument('--hop_length', type=int, default=512, help='Hop Length')
parser.add_argument('--n_mels', type=int, default=256, help='Number of Mel Bands')
parser.add_argument('--n_mfcc', type=int, default=30, help='Number of MFCCs')
parser.add_argument('--split_method', type=int, required=True, help='Data split method')
parser.add_argument('--model_name', type=str, required=True, help='Model name')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience') 
parser.add_argument('--repeat', type=int, help='Repeat number')  
args = parser.parse_args()

os.makedirs(args.checkpoints_path, exist_ok=True)
os.makedirs(args.result_path, exist_ok=True)
os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

# Set up logging
def setup_logging(model_name, feature_type, split_method, repeat):
    log_file_path = os.path.join(args.log_path, f"{model_name}_{feature_type}_split{split_method}_repeat{repeat}.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    return log_file_path

log_file_path = setup_logging(args.model_name, args.feature_type, args.split_method, args.repeat)

message = '----------------- Options ---------------\n'
for k, v in sorted(vars(args).items()):
    comment = ''
    default = parser.get_default(k)
    if v != default:
        comment = '\t[default: %s]' % str(default)
    message += '{:>20}: {:<15}{}\n'.format(str(k), str(v), comment)
message += '----------------- End -------------------'
print(message)
logging.info(message)

print(f"Logging to {log_file_path}")
logging.info(f"Logging initialized. Saving logs to {log_file_path}")

def load_wav_files(path):
    files = glob.glob(f'{path}/**/*.wav', recursive=True)
    logging.info(f"Loaded {len(files)} files from {path}")
    return files

def sample_files(src_dirs, num_samples):
    all_files = []
    for src_dir in src_dirs:
        files = load_wav_files(src_dir)
        all_files.extend(files)
    return np.random.choice(all_files, min(num_samples, len(all_files)), replace=False)

train_nonspam_path = 'normal_normal_normal_path'
train_spam1_path = 'normal_vishing_normal_path'
train_spam2_path = 'vishing_normal_vishing_path'
spam_path = 'original_vishing_path'
nonspam_path = 'original_nonvishing_true_data_path'

if args.split_method == 1:
    normal_sample = sample_files([train_nonspam_path], 3000)
    spam_sample = sample_files([train_spam1_path, train_spam2_path], 3000)
    paths = np.concatenate([normal_sample, spam_sample])
    labels = np.concatenate([np.zeros(len(normal_sample)), np.ones(len(spam_sample))])
    X_train, X_test, y_train, y_test = train_test_split(paths, labels, stratify=labels, test_size=0.2, random_state=42)
    split_name = "Method 1: Concat only binary classification"

elif args.split_method == 2:
    spam_sample = sample_files([spam_path], 3000)
    nonspam_sample = sample_files([nonspam_path], 3000)
    paths = np.concatenate([spam_sample, nonspam_sample])
    labels = np.concatenate([np.ones(len(spam_sample)), np.zeros(len(nonspam_sample))])
    X_train, X_test, y_train, y_test = train_test_split(paths, labels, stratify=labels, test_size=0.2, random_state=42)
    split_name = "Method 2: Basic only binary classification"

elif args.split_method == 3:
    train_spam = sample_files([train_spam1_path, train_spam2_path], 2400)
    train_nonspam = sample_files([train_nonspam_path], 2400)
    test_spam = sample_files([spam_path], 300)
    test_nonspam = sample_files([nonspam_path], 300)

    X_train = np.concatenate([train_spam, train_nonspam])
    y_train = np.concatenate([np.ones(len(train_spam)), np.zeros(len(train_nonspam))])
    X_test = np.concatenate([test_spam, test_nonspam])
    y_test = np.concatenate([np.ones(len(test_spam)), np.zeros(len(test_nonspam))])

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    split_name = "Method 3: Concat train, basic test"

elif args.split_method == 4:
    test_spam = sample_files([train_spam1_path, train_spam2_path], 300)
    test_nonspam = sample_files([train_nonspam_path], 300)

    train_spam = sample_files([spam_path], 2400)
    train_nonspam = sample_files([nonspam_path], 2400)

    X_train = np.concatenate([train_spam, train_nonspam])
    y_train = np.concatenate([np.ones(len(train_spam)), np.zeros(len(train_nonspam))])
    X_test = np.concatenate([test_spam, test_nonspam])
    y_test = np.concatenate([np.ones(len(test_spam)), np.zeros(len(test_nonspam))])

    split_name = "Method 4: Basic train, concat test"

elif args.split_method == 5:
    train_spam = sample_files([train_spam1_path, train_spam2_path, spam_path], 2400)
    train_nonspam = sample_files([train_nonspam_path, nonspam_path], 2400)
    test_spam = sample_files([train_spam1_path, train_spam2_path, spam_path], 300)
    test_nonspam = sample_files([train_nonspam_path, nonspam_path], 300)

    X_train = np.concatenate([train_spam, train_nonspam])
    y_train = np.concatenate([np.ones(len(train_spam)), np.zeros(len(train_nonspam))])
    X_test = np.concatenate([test_spam, test_nonspam])
    y_test = np.concatenate([np.ones(len(test_spam)), np.zeros(len(test_nonspam))])

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    split_name = "Method 5: All combined"

else:
    raise ValueError("Invalid split method selected.")

print(f'Split method: {split_name}')
logging.info(f'Split method: {split_name}')

def SetData(paths, labels, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type):
    dataset = []
    labels_data = []
    for idx, path in enumerate(paths):
        y, sr = librosa.load(path)
        if feature_type == 'stft':
            D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
            tmp = librosa.power_to_db(D, ref=np.max)
        elif feature_type == 'mel':
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
            tmp = librosa.power_to_db(mel_spec, ref=np.max)
        elif feature_type == 'mfcc':
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            tmp = mfcc
        tmp = tmp.flatten()
        dataset.append(tmp)
        labels_data.append(int(labels[idx]))
    
    return np.array(dataset), np.array(labels_data)

trainset, train_label = SetData(X_train, y_train, args.n_fft, args.win_length, args.hop_length, args.n_mels, args.n_mfcc, args.feature_type)
testset, test_label = SetData(X_test, y_test, args.n_fft, args.win_length, args.hop_length, args.n_mels, args.n_mfcc, args.feature_type)

models = {
    'Logistic': LogisticRegression(max_iter=100, C=100, penalty='l2', solver='lbfgs'),
    'DT': DecisionTreeClassifier(criterion='entropy', max_depth=10),
    'RF': RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=100),
    'SVM': SVC(C=10, kernel='linear')
}

early_stopping_patience = args.patience

for model_name, model in models.items():
    setup_logging(model_name, args.feature_type, args.split_method,args.repeat)

    csv_file_path = os.path.join(args.result_path, f"{model_name}_{args.feature_type}_split{args.split_method}.csv")

    logging.info(f"Training {model_name} model...")
    best_accuracy = 0
    no_improvement_count = 0
    start = time.time()
    for epoch in range(args.num_epoch):
        model.fit(trainset, train_label)
        train_time = time.time() - start
        pred = model.predict(trainset)
        train_accuracy = np.mean(pred == train_label)
        print(f"Epoch {epoch+1}/{args.num_epoch}, Training accuracy: {train_accuracy:.4f}, Time: {train_time:.2f}s")
        logging.info(f"Epoch {epoch+1}/{args.num_epoch}, Training accuracy: {train_accuracy:.4f}, Time: {train_time:.2f}s")

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

    start_test = time.time()
    pred = model.predict(testset)
    test_time = time.time() - start_test
    test_accuracy = np.mean(pred == test_label)
    print(f"{model_name} test accuracy: {test_accuracy:.4f}, Test time: {test_time:.2f}s")
    logging.info(f"{model_name} test accuracy: {test_accuracy:.4f}, Test time: {test_time:.2f}s")

    report = classification_report(test_label, pred, digits=6, output_dict=True)
    result = pd.DataFrame(report).transpose()
    result['train_time'] = train_time
    result['test_time'] = test_time
    result['model'] = model_name

    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    result.to_csv(csv_file_path)

print("FINISH")
logging.info("FINISH")
