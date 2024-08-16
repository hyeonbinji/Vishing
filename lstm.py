import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import glob
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import time
import torch.optim as optim
from sklearn.metrics import classification_report
import logging

# Argument parsing
parser = argparse.ArgumentParser(description="PyTorch LSTM Training")
parser.add_argument('--checkpoints_path', type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--result_path', type=str, required=True, help='Path to save results')
parser.add_argument('--log_path', type=str, required=True, help='Path to save logs')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--feature_type', type=str, default='stft', help='Feature type (stft, mel, mfcc)')
parser.add_argument('--feature_time', type=float, default=0.5, help='Feature time window size')
parser.add_argument("--n_fft", default=4096, type=int, help="Number of FFT components")
parser.add_argument("--win_length", default=4096, type=int, help="Window length")
parser.add_argument("--hop_length", default=512, type=int, help="Hop length")
parser.add_argument("--n_mels", default=256, type=int, help="Number of Mel bands")
parser.add_argument("--n_mfcc", default=30, type=int, help="Number of MFCCs")
parser.add_argument('--num_test', type=int, default=0, help='Test number')
parser.add_argument('--split_method', type=int, default=1, help='Data split method (1, 2, 3, 4, 5)')
parser.add_argument('--repeat', type=int, default=1, help='Repeat number')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
args = parser.parse_args()

# Logging setup
log_file_path = os.path.join(args.log_path, f'LSTM_{args.feature_type}_split{args.split_method}_repeat{args.repeat}.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

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

# 데이터 로드 함수
def load_wav_files(path):
    files = glob.glob(f'{path}/**/*.wav', recursive=True)
    print(f"Loaded {len(files)} files from {path}")
    logging.info(f"Loaded {len(files)} files from {path}")
    return files

def sample_files(src_dirs, num_samples):
    all_files = []
    for src_dir in src_dirs:
        files = load_wav_files(src_dir)
        all_files.extend(files)
    sampled_files = np.random.choice(all_files, min(num_samples, len(all_files)), replace=False)
    print(f"Sampled {len(sampled_files)} files")
    logging.info(f"Sampled {len(sampled_files)} files")
    return sampled_files

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

# 데이터 준비 함수
def SetData(paths, labels, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type):
    dataset = []
    target_dim = 173
    for idx, path in enumerate(paths):
        y, sr = librosa.load(path, sr=None)
        D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        
        if feature_type == 'stft':
            tmp = librosa.power_to_db(D, ref=np.max)
        elif feature_type == 'mel':
            mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
            tmp = librosa.amplitude_to_db(mel_spec, ref=0.00002)
        elif feature_type == 'mfcc':
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(D), sr=sr, n_mfcc=n_mfcc)
            tmp = librosa.amplitude_to_db(mfcc, ref=0.00002)
        
        if tmp.shape[1] > target_dim:
            tmp = tmp[:, :target_dim]
        elif tmp.shape[1] < target_dim:
            padding = target_dim - tmp.shape[1]
            tmp = np.pad(tmp, ((0, 0), (0, padding)), mode='constant')

        tmp = np.expand_dims(tmp, axis=0)
        tmp = torch.tensor(tmp)
        tmp_y = int(labels[idx])
        dataset.append((tmp, tmp_y))
    return dataset

# LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self, device, input_dim, hidden_dim):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.build_model()
        self.to(device)
    
    def build_model(self):
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=3, dropout=0.2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.BCE_loss = nn.BCELoss()

    def forward(self, x):
        if x.dim() == 3:
            output, _ = self.lstm(x)
            output = torch.mean(output, dim=1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
            output, _ = self.lstm(x)
            output = torch.mean(output, dim=1)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        output = self.fc(output)
        if output.dim() == 3:
            output = output.squeeze(1)
        elif output.dim() == 2:
            output = output.squeeze(1)
        
        output = self.sigmoid(output)
        return output

    def train(self, train_loader, valid_loader, epochs, learning_rate, patience=10):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        best_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False

        loss_log = []
        for e in range(epochs):
            epoch_loss = 0
            for _, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                data = data.squeeze()
                data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
                out = self.forward(data)
                loss = self.BCE_loss(out, target)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            loss_log.append(epoch_loss)
            valid_acc, valid_loss = self.predict(valid_loader)
            logging.info(f'>> [Epoch {e+1}/{epochs}] Total epoch loss: {epoch_loss:.2f} / Valid accuracy: {100*valid_acc:.2f}% / Valid loss: {valid_loss:.4f}')
            print(f'>> [Epoch {e+1}/{epochs}] Total epoch loss: {epoch_loss:.2f} / Valid accuracy: {100*valid_acc:.2f}% / Valid loss: {valid_loss:.4f}')
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    early_stop = True
                    print(f"Early stopping at epoch {e+1}")
                    logging.info(f"Early stopping at epoch {e+1}")
                    break
        
        return loss_log
    
    def predict(self, valid_loader, return_preds=False):
        BCE_loss = nn.BCELoss(reduction='sum')
        preds = []
        total_loss = 0
        correct = 0
        len_data = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(valid_loader):
                data = data.squeeze()
                data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
                out = self.forward(data)
                len_data += len(target)
                loss = BCE_loss(out, target)
                total_loss += loss
                pred = (out > 0.5).detach().cpu().numpy().astype(np.float32)
                preds += list(pred)
                correct += sum(pred == target.detach().cpu().numpy())
            acc = correct / len_data
            loss = total_loss / len_data
        if return_preds:
            return acc, loss, preds
        else:
            return acc, loss


# Paths
train_nonspam_path = 'normal_normal_normal_path'
train_spam1_path = 'normal_vishing_normal_path'
train_spam2_path = 'vishing_normal_vishing_path'
spam_path = 'original_vishing_path'
nonspam_path = 'original_nonvishing_true_data_path'

               
def split_data():
    if args.split_method == 1:
        normal_sample = sample_files([train_nonspam_path], 3000)
        spam_sample = sample_files([train_spam1_path, train_spam2_path], 3000)
        paths = np.concatenate([normal_sample, spam_sample])
        labels = np.concatenate([np.zeros(len(normal_sample)), np.ones(len(spam_sample))])
        split_name = "Method 1: Concat only binary classification"

    elif args.split_method == 2:
        spam_sample = sample_files([spam_path], 3000)
        nonspam_sample = sample_files([nonspam_path], 3000)
        paths = np.concatenate([spam_sample, nonspam_sample])
        labels = np.concatenate([np.ones(len(spam_sample)), np.zeros(len(nonspam_sample))])
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

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
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
        split_name = "Method 5: All combined"

    else:
        raise ValueError("Invalid split method selected.")

    if args.split_method in [1, 2]:
        X_train, X_test, y_train, y_test = train_test_split(paths, labels, stratify=labels, test_size=0.2, random_state=42)

    # Validation set split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.16, random_state=42)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, split_name

X_train, X_valid, X_test, y_train, y_valid, y_test, split_name = split_data()

# Dataset and DataLoader setup
trainset = SetData(X_train, y_train, args.n_fft, args.win_length, args.hop_length, args.n_mels, args.n_mfcc, args.feature_type)
validset = SetData(X_valid, y_valid, args.n_fft, args.win_length, args.hop_length, args.n_mels, args.n_mfcc, args.feature_type)
testset = SetData(X_test, y_test, args.n_fft, args.win_length, args.hop_length, args.n_mels, args.n_mfcc, args.feature_type)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=0)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)


# 모델 학습 및 평가
input_dim = trainset[0][0].shape[-1]
hidden_dim = 16
model = LSTM(device=device, input_dim=input_dim, hidden_dim=hidden_dim)
model.to(device)

start = time.time()
model.train(train_loader=train_loader, valid_loader=valid_loader, epochs=args.num_epoch, learning_rate=args.learning_rate, patience=args.patience)
train_time = time.time() - start
logging.info(f'Train time: {train_time}')
print(f'Train time: {train_time}')
torch.save(model.state_dict(), os.path.join(args.checkpoints_path, f'LSTM_{args.feature_time}_{args.num_test}.pt'))

start_t = time.time()
acc, loss, p = model.predict(test_loader, return_preds=True)
test_time = time.time() - start_t
logging.info(f'Test time: {test_time}')
print(f'Test time: {test_time}')

report = classification_report(y_test, p, digits=6, output_dict=True)

result = pd.DataFrame(report).transpose()
result['train_time'] = train_time
result['test_time'] = test_time
result.to_csv(os.path.join(args.result_path, f'LSTM_{args.feature_type}_split{args.split_method}_repeat{args.repeat}.csv'))

logging.info(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Accuracy: {acc*100:.2f}%")
logging.info("FINISH")
print("FINISH")
