# coding=utf-8

import argparse
import time
import os
import logging
import yaml
import datetime
import random

from dataset import *
from model import *
from metrics import batch_performance
from utils import *

# clear cache
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="IQON help='IQON/POLYVORE/ALIBABA')
parser.add_argument('--seed', default=2023, help='Random seed')
parser.add_argument('--attr', default=2.5, type=float, help='attr 2.5 or 100')
parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--deviceID', type=int, default=0)
parser.add_argument('--lambda_cl', type=float, default=0.1, help='lambda of contrastive loss')
parser.add_argument('--num_user_layers', type=int, default=3)
parser.add_argument('--num_outfit_layers', type=int, default=3)
parser.add_argument('--disen', type=int, default=3,
                    help='layer number of directed hypergraph convolutional network')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--keep_rate', type=float, default=1, help='ratio of edges to keep')
parser.add_argument('--keep_rate_item', type=float, default=1, help='ratio of item-item directed edges to keep')  # 0.7
parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='Learning rate scheduler factor')
parser.add_argument('--save_dir', type=str, default="logs")
parser.add_argument("--saved_model_path", type=str, default="20240118_231414")
args = parser.parse_args()

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# set device gpu/cpu
device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu")

# set save_dir
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
current_save_dir = os.path.join(args.save_dir, args.saved_model_path)
if not os.path.exists(current_save_dir):
    # create current save_dir
    os.mkdir(current_save_dir)

# Setup logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(current_save_dir, f"log_training.txt"),
                    filemode='w+')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.getLogger('matplotlib.font_manager').disabled = True

# Save run settings
args_filename = args.dataset + '_inference_args.yaml'
with open(os.path.join(current_save_dir, args_filename), 'w') as f:
    yaml.dump(vars(args), f, sort_keys=False)


def main():
    # Parse Arguments
    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: {}".format(device))
    if args.dataset == "POLYVORE":
        NUM_USERS = 2173
        NUM_ITEMS = 7038
        PADDING_IDX = NUM_ITEMS
    elif args.dataset == "IQON":
        NUM_USERS = 834
        NUM_ITEM = 3835
        PADDING_IDX = NUM_ITEMS

    # Load Dataset
    logging.info("2. Load Test Dataset")
    test_dataset = ITEMDataset(data_filename="datasets/{}/test_item.txt".format(args.dataset),
                              outfit_filename="datasets/{}/{}_items_item.pkl".format(args.dataset,
                                                                                                args.dataset),
                              num_users=NUM_USERS, num_items=NUM_ITEMS, padding_idx=PADDING_IDX,
                              args=args, device=device)

    active_user_dict = load_dict_from_pkl("datasets/{}/active_user_dict.pkl".format(args.dataset))
    active_user_indices = active_user_dict.keys()
    test_partial_dataset = ITEMPartialDataset(test_dataset, user_indices=active_user_indices)
    logging.info("active user inference")

    # 3. Construct DataLoader
    logging.info("3. Construct DataLoader")
    test_dataloader = DataLoader(dataset=test_partial_dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))

    # Load Model
    logging.info("4. Load Model")
    model = PCM(NUM_USERS, NUM_ITEMS, args, device)

    # load state dict
    model_state_dict = torch.load(os.path.join(current_save_dir, "{}.pt".format(args.dataset)))

    # load state_dict to model
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # Inference
    logging.info("5. Start Inference")
    Ks_list = [ 5, 10]
    final_results = { "HR5": 0.0, "HR10": 0.0, 
                      "NDCG5": 0.0, "NDCG10": 0.0, }

    logging.info("Testing")
    test_loss = 0.0
    test_hr_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))
    test_ndcg_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):

            logging.info("Test. Batch {}/{}".format(idx, len(test_dataloader)))

            predictions, loss_cl_users, loss_cl_items = model(test_dataset, batch)

            for k in Ks_list:
                hr, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                col_idx = Ks_list.index(k)
                test_hr_array[idx, col_idx] = hr
                test_ndcg_array[idx, col_idx] = ndcg

    logging.info("Testing finishes")
    logging.info("Testing loss: {}".format(test_loss / len(test_dataloader)))
    logging.info("Testing results:")
    for k in Ks_list:
        col_idx = Ks_list.index(k)
        hr = np.mean(test_hr_array[:, col_idx])
        ndcg = np.mean(test_ndcg_array[:, col_idx])
        logging.info("HR@{}: {:.4f}".format(k, hr))
        logging.info("NDCG@{}: {:.4f}".format(k, ndcg))

    # update best result
    for k in Ks_list:
        if k == 1:
            final_results["HR1"] = max(final_results["HR1"], np.mean(test_hr_array[:, 0]))
            final_results["NDCG1"] = max(final_results["NDCG1"], np.mean(test_ndcg_array[:, 0]))

        elif k == 5:
            final_results["HR5"] = max(final_results["HR5"], np.mean(test_hr_array[:, 1]))
            final_results["NDCG5"] = max(final_results["NDCG5"], np.mean(test_ndcg_array[:, 1]))

        elif k == 10:
            final_results["HR10"] = max(final_results["HR10"], np.mean(test_hr_array[:, 2]))
            final_results["NDCG10"] = max(final_results["NDCG10"], np.mean(test_ndcg_array[:, 2]))

        elif k == 20:
            final_results["HR20"] = max(final_results["HR20"], np.mean(test_hr_array[:, 3]))
            final_results["NDCG20"] = max(final_results["NDCG20"], np.mean(test_ndcg_array[:, 3]))
    logging.info("==================================\n\n")

    logging.info("6. Final Results")
    formatted_dict = {key: f"{value:.4f}" for key, value in final_results.items()}
    logging.info(formatted_dict)
    logging.info("\n")


if __name__ == '__main__':
    main()
