import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_names', type=str, nargs='+', default=['Logistic', 'DT', 'RF', 'SVM'], help='List of model names')
parser.add_argument('--feature_types', type=str, nargs='+', default=['stft','mfcc', 'stft'], help='List of feature types')
parser.add_argument('--split_methods', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='List of split methods')
parser.add_argument('--checkpoints_path', type=str, required=True, help='Path to checkpoints')
parser.add_argument('--result_path', type=str, required=True, help='Path to results')
parser.add_argument('--log_path', type=str, required=True, help='Path to logs')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_test', type=int, default=0, help='Number of tests')
parser.add_argument('--feature_time', type=float, default=0.5, help='Feature Time Size')
parser.add_argument('--num_repeats', type=int, default=5, help='Number of repetitions for each combination')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

args = parser.parse_args()

# Ensure directories exist
os.makedirs(args.result_path, exist_ok=True)
os.makedirs(args.checkpoints_path, exist_ok=True)
os.makedirs(args.log_path, exist_ok=True)

basic_script_path = "basic.py"

# Iterate over all combinations of split_methods, model_names, and feature_types
for split_method in args.split_methods:
    for model_name in args.model_names:
        for feature_type in args.feature_types:
            for repeat in range(1, args.num_repeats + 1):
                result_model_path = os.path.join(args.result_path, f'{split_method}', model_name, feature_type, f'repeat{repeat}')
                checkpoints_model_path = os.path.join(args.checkpoints_path, f'{split_method}', model_name, feature_type, f'repeat{repeat}')
                log_model_path = os.path.join(args.log_path, f'{split_method}', model_name, feature_type, f'repeat{repeat}')
                
                os.makedirs(result_model_path, exist_ok=True)
                os.makedirs(checkpoints_model_path, exist_ok=True)
                os.makedirs(log_model_path, exist_ok=True)

                command = (
                    f'python {basic_script_path} --model_name {model_name} '
                    f'--feature_time {args.feature_time} --feature_type {feature_type} '
                    f'--checkpoints_path {checkpoints_model_path} --result_path {result_model_path} '
                    f'--log_path {log_model_path} '
                    f'--gpu_id {args.gpu_id} --split_method {split_method} --num_test {args.num_test} --repeat {repeat} '
                    f'--patience {args.patience}'
                )

                os.system(command)

print("FINISH")
