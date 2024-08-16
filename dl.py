import os
import argparse
import logging

# Argument parsing
parser = argparse.ArgumentParser(description="Run multiple models with different settings")
parser.add_argument('--model_name', type=str, default='DenseNet', help='Model name (DenseNet or LSTM)')
parser.add_argument('--checkpoints_path', type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--result_path', type=str, required=True, help='Path to save results')
parser.add_argument('--log_path', type=str, required=True, help='Path to save logs')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--num_repeats', type=int, default=5, help='Number of repetitions for each combination')
parser.add_argument('--feature_types', type=str, nargs='+', default=['mfcc', 'mel', 'stft'], help='List of feature types')  # 추가된 부분
parser.add_argument('--split_methods', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='List of split methods')  # 추가된 부분
args = parser.parse_args()

# Set up logging
log_file_path = os.path.join(args.log_path, 'dl_training_log.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting training with the following parameters:")
logging.info(f"Model: {args.model_name}")
logging.info(f"GPU ID: {args.gpu_id}")
logging.info(f"Patience: {args.patience}")
logging.info(f"Number of Repetitions: {args.num_repeats}")
logging.info(f"Feature Types: {args.feature_types}")
logging.info(f"Split Methods: {args.split_methods}")

# Loop over all combinations
for i in range(1, args.num_repeats + 1):  
    for feature_type in args.feature_types:
        for split_method in args.split_methods:
            command = (
                f'python /path/{args.model_name}.py '
                f'--feature_time 0.5 --feature_type {feature_type} '
                f'--checkpoints_path {args.checkpoints_path} --result_path {args.result_path} '
                f'--log_path {args.log_path} --gpu_id {args.gpu_id} '
                f'--split_method {split_method} '
                f'--repeat {i} --patience {args.patience}'
            )
            logging.info(f"Executing command: {command}")
            os.system(command)
            logging.info(f"Finished training for feature type {feature_type}, split method {split_method}, repeat {i}")

logging.info("All training jobs completed.")
print("FINISH")
