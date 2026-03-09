# coding=utf-8


import argparse
import time
import os
import logging
import yaml
import datetime
import torch.optim as optim
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
parser.add_argument('--dataset', default="IQON", help='IQON/POLYVORE')
parser.add_argument('--seed', default=2023, help='Random seed')
parser.add_argument('--attr', default=2.5, type=float, help='attr 2.5 or 0.25')
parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size')
parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--decay', type=float, default=5e-4)    # 5e-4
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')    # 0.3
parser.add_argument('--deviceID', type=int, default=0)
parser.add_argument('--lambda_cl', type=float, default=0.1, help='lambda of contrastive loss')
parser.add_argument('--num_user_layers', type=int, default=3)
parser.add_argument('--num_outfit_layers', type=int, default=3)
parser.add_argument('--disen', type=int, default=3, help='layer number of directed hypergraph convolutional network')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--keep_rate', type=float, default=1, help='ratio of edges to keep')
parser.add_argument('--keep_rate_item', type=float, default=1, help='ratio of item-item directed edges to keep')  # 0.7
parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='Learning rate scheduler factor')
parser.add_argument('--save_dir', type=str, default="logs")
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
current_save_dir = os.path.join(args.save_dir, current_time)

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
args_filename = args.dataset + '_args.yaml'
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
        NUM_ITEMS = 3835
        PADDING_IDX = NUM_ITEMS

    # Load Dataset
    logging.info("2. Load Dataset")
    train_dataset = ITEMDataset(data_filename="datasets/{}/train_item.txt".format(args.dataset),
                               outfit_filename="datasets/{}/{}_items_item.pkl".format(args.dataset, args.dataset),
                               num_users=NUM_USERS,
                               num_items=NUM_ITEMS,
                               padding_idx=PADDING_IDX,
                               args=args,
                               device=device)

    test_dataset = ITEMDataset(data_filename="datasets/{}/test_item.txt".format(args.dataset),
                              outfit_filename="datasets/{}/{}_items_item.pkl".format(args.dataset, args.dataset),
                              num_users=NUM_USERS,
                              num_items=NUM_ITEMS,
                              padding_idx=PADDING_IDX,
                              args=args,
                              device=device)

    # 3. Construct DataLoader
    logging.info("3. Construct DataLoader")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))

    # Load Model
    logging.info("4. Load Model")
    model = PCM(NUM_USERS, NUM_ITEMS, args, device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # Train
    logging.info("5. Start Training")
    Ks_list = [1, 5, 10, 20]
    final_results = {"HR1": 0.0, "HR5": 0.0, "HR10": 0.0, "HR20": 0.0,
                     "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0, "NDCG20": 0.0,
                     }

    monitor_loss = float('inf')
    best_test_rec5 = 0.0
    for epoch in range(args.num_epochs):
        logging.info("================= Epoch {}/{} =================".format(epoch, args.num_epochs))
        start_time = time.time()
        model.train()

        train_loss = 0.0

        # to save hr and ndcg results
        train_hr_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        train_ndcg_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        for idx, batch in enumerate(train_dataloader):
            logging.info("Train. Batch {}/{}".format(idx, len(train_dataloader)))
            optimizer.zero_grad()

            predictions, loss_cl_users, loss_cl_items = model(train_dataset, batch)

            # calculate loss
            loss_rec = criterion(predictions, batch["label"].to(device))
            loss = loss_rec + args.lambda_cl * (loss_cl_items + loss_cl_users)
            logging.info("Train. loss_rec: {:.4f}; loss_cl_items: {:.4f}; loss_cl_users: {:.4f}; "
                         "loss: {:.4f}".format(loss_rec.item(), loss_cl_items, loss_cl_users, loss))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            for k in Ks_list:
                hr, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                col_idx = Ks_list.index(k)
                train_hr_array[idx, col_idx] = hr
                train_ndcg_array[idx, col_idx] = ndcg

        logging.info("Training finishes at this epoch. It takes {} min".format((time.time() - start_time) / 60))
        logging.info("Training loss: {:.4f}".format(train_loss / len(train_dataloader)))
        logging.info("Training Epoch {}/{} results:".format(epoch, args.num_epochs))
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            logging.info("HR@{}: {:.4f}".format(k, np.mean(train_hr_array[:, col_idx])))
            logging.info("NDCG@{}: {:.4f}".format(k, np.mean(train_ndcg_array[:, col_idx])))
        logging.info("\n")

        logging.info("Testing")
        test_loss = 0.0
        test_hr_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))
        test_ndcg_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):

                logging.info("Test. Batch {}/{}".format(idx, len(test_dataloader)))

                predictions, loss_cl_users, loss_cl_items = model(test_dataset, batch)

                # calculate loss
                loss_rec = criterion(predictions, batch["label"].to(device))
                loss = loss_rec + args.lambda_cl * (loss_cl_items + loss_cl_users)
                logging.info("Test. loss_rec: {:.4f}; loss_cl_items: {:.4f}; loss_cl_users: {:.4f}; "
                             "loss: {:.4f}".format(loss_rec.item(), loss_cl_items, loss_cl_users, loss))

                test_loss += loss.item()

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

        # Check monitor loss and monitor score for updating
        monitor_loss = min(monitor_loss, test_loss)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

      
        test_hr5 = np.mean(test_hr_array[:, 1])
        if test_hr5 > best_test_rec5:
            best_test_rec5 = test_hr5
            logging.info("Update test results and save model at epoch{}".format(epoch))

            # define saved_model_path
            saved_model_path = os.path.join(current_save_dir, "{}.pt".format(args.dataset))
            torch.save(model.state_dict(), saved_model_path)

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

