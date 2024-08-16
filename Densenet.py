import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import glob
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import time
import torch.optim as optim
from sklearn.metrics import classification_report
import copy

# Argument parsing
parser = argparse.ArgumentParser(description="PyTorch DenseNet Training")
parser.add_argument('--checkpoints_path', type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--result_path', type=str, required=True, help='Path to save results')
parser.add_argument('--log_path', type=str, required=True, help='Path to save logs')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_test', type=int, default=0, help='Test number')
parser.add_argument('--feature_type', type=str, default='stft', help='Feature type (stft, mel, mfcc)')
parser.add_argument('--feature_time', type=float, default=0.5, help='Feature time window size')
parser.add_argument("--n_fft", default=4096, type=int, help="Number of FFT components")
parser.add_argument("--win_length", default=4096, type=int, help="Window length")
parser.add_argument("--hop_length", default=512, type=int, help="Hop length")
parser.add_argument("--n_mels", default=256, type=int, help="Number of Mel bands")
parser.add_argument("--n_mfcc", default=30, type=int, help="Number of MFCCs")
parser.add_argument('--split_method', type=int, default=1, help='Data split method (1, 2, 3, 4, 5)')
parser.add_argument('--repeat', type=int, default=1, help='Repeat number')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
args = parser.parse_args()

log_file_path = os.path.join(args.log_path, f'DenseNet_{args.feature_type}_split{args.split_method}_repeat{args.repeat}.log')
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

# Data Preparation Functions
def load_wav_files(path):
    files = glob.glob(f'{path}/**/*.wav', recursive=True)
    logging.info(f"Loaded {len(files)} files from {path}")
    return files

def sample_files(src_dirs, num_samples):
    all_files = []
    for src_dir in src_dirs:
        files = load_wav_files(src_dir)
        all_files.extend(files)
    sampled_files = np.random.choice(all_files, min(num_samples, len(all_files)), replace=False)
    logging.info(f"Sampled {len(sampled_files)} files")
    return sampled_files

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

def SetData(paths, labels, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type):
    dataset = []
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
        tmp = np.expand_dims(tmp, axis=0)
        tmp = torch.tensor(tmp, dtype=torch.float32)
        tmp_y = int(labels[idx])
        dataset.append((tmp, tmp_y))
    return dataset

# Model definition
class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channels)
        self.conv2 = nn.Conv2d(inner_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out

class DenseNet(nn.Module):
    def __init__(self, growth_rate, nblocks, reduction, num_classes, init_weights=True):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, inner_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.features = nn.Sequential()

        for i in range(len(nblocks) - 1):
            self.features.add_module('dense_block_{}'.format(i), self._make_dense_block(nblocks[i], inner_channels))
            inner_channels += self.growth_rate * nblocks[i]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_{}'.format(i), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module('dense_block_{}'.format(len(nblocks) - 1), self._make_dense_block(nblocks[len(nblocks) - 1], inner_channels))
        inner_channels += self.growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.sigmoid(out)
        return out

    def _make_dense_block(self, nblock, inner_channels):
        layers = []
        for _ in range(nblock):
            layers.append(BottleNeck(inner_channels, self.growth_rate))
            inner_channels += self.growth_rate
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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

# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)
    
    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    
    return loss_b.item(), metric_b                

# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = 0

    for xb, yb in dataset_dl:
        xb = torch.tensor(xb)
        xb = xb.to(device)
        yb = torch.tensor(yb)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        len_data += len(yb)
        
        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break
    
    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric

# function to start training
def train_val(model, params, patience=10):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    epochs_no_improve = 0

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch+1, num_epochs, current_lr))
        logging.info('Epoch {}/{}, current lr= {}'.format(epoch+1, num_epochs, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, val_metric*100, (time.time()-start_time)/60))
        print('-'*10)
        logging.info('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, val_metric*100, (time.time()-start_time)/60))

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            logging.info(f'Early stopping at epoch {epoch+1}')
            break

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history

# Model initialization
model = DenseNet(growth_rate=12, nblocks=[6,12,24,6], reduction=0.5, num_classes=2)
model.to(device)

# Training parameters
opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=8)

params_train = {
    'num_epochs': args.num_epoch,
    'optimizer': opt,
    'loss_func': nn.CrossEntropyLoss(reduction='sum'),
    'train_dl': train_loader,
    'val_dl': valid_loader,
    'lr_scheduler': lr_scheduler,
    'path2weights': os.path.join(args.checkpoints_path, f'DenseNet_{args.feature_type}_split{args.split_method}.pt'),
    'sanity_check': False
}

# Training the model
model, loss_hist, metric_hist = train_val(model, params_train, patience=args.patience)
# Test after training completes
model.eval()
pred = []
start_time = time.time()
for data, _ in test_loader:
    data = data.to(device)
    output = model(data)
    output = output.detach().cpu().numpy()
    output = output.argmax(axis=1)
    pred.append(output)

test_time = time.time() - start_time
report = classification_report(y_test, np.array(pred).flatten(), digits=6, output_dict=True)

result = pd.DataFrame(report).transpose()
result['train_time'] = sum([h for h in loss_hist['train']])
result['test_time'] = test_time
result_file = os.path.join(args.result_path, f'DenseNet_{args.feature_type}_split{args.split_method}_repeat{args.repeat}.csv')
result.to_csv(result_file)

# Log test accuracy
test_accuracy = np.mean(np.array(pred).flatten() == y_test)
logging.info(f"Test Accuracy: {test_accuracy*100:.2f}%")
logging.info(f"Test Time: {test_time:.4f} seconds")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
