"""
Main function for conformal prediction using zero-shot, Conf-OT and baseline methods for transfer learning.
It includes three non-conformity scores: LAC, APS, and RAPS.
"""

import argparse
import torchsort
import os
import conformal
import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from conformal.metrics import evaluate_conformal, accuracy
from solvers.transductive import confot, TransCLIP, TIM
from solvers.inductive import Linear

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')

def process(args):
    res = pd.DataFrame()
    results_detailed = {}

    # Các DataFrame lưu cho biểu đồ
    coverage_list = []
    time_list = []
    avg_size_list = []
    delta_list = []

    for i_domain in range(0, len(args.test_datasets)):
        print("  Testing on: [{dataset}]".format(dataset=args.test_datasets[i_domain]))

        id = "local_data/confot-09-22_23-28-46.npy"
        if os.path.isfile(id):
            print("  Loading features from npy file")
            cache = np.load(id, allow_pickle=True)
            if isinstance(cache, dict):
                logits_ds = torch.tensor(cache["logits_ds"])
                labels_ds = torch.tensor(cache["refs_ds"]).to(torch.long)
            elif isinstance(cache, (list, tuple)):
                logits_ds = torch.tensor(cache[0])
                labels_ds = torch.tensor(cache[1]).to(torch.long)
            else:
                raise ValueError("File npy không đúng định dạng!")
        else:
            continue


        emp_cov, set_size, strat_covgap, class_covgap = [], [], [], []
        top1, top5 = [], []
        time_adapt, time_conf_fit, time_conf_inf = [], [], []

        for _ in tqdm(range(args.seeds), leave=False, desc="  Conformal inference: "):
            torch.cuda.empty_cache()

            logits_calib, labels_calib, logits_test, labels_test = conformal.split_data(logits_ds, labels_ds, p=args.p)
            logits_ds, labels_ds = torch.cat([logits_calib, logits_test]), torch.cat([labels_calib, labels_test])

            # Conf-OT transductive transfer learning approach
            time_adapt_i_1 = time.time()
            if args.adapt == "confot":
                z = confot.compute_codes(logits_ds, epsilon=args.epsilon, num_iters=args.ot_iters,
                                         observed_marginal=args.observed_marginal,
                                         labels_count=np.bincount(labels_calib)).to("cpu")
            elif args.adapt == "tim":
                that = 100
                z, that = TIM.compute_codes(logits_ds, observed_marginal=args.observed_marginal,
                                            labels_count=np.bincount(labels_calib), that=that, hp_search=(_ == 0),
                                            labels_calib=labels_calib)
            elif args.adapt == "transclip":
                z = TransCLIP.compute_codes(logits_ds, labels_ds)
            elif args.adapt == "linear_probe":
                z, that = Linear.compute_codes(logits_calib, labels_calib, logits_test, hp_search=(_ == 0))
            else:
                z = torch.softmax(torch.tensor(logits_ds)/args.epsilon, dim=-1)

            time_adapt_i_2 = time.time()
            time_adapt_i = time_adapt_i_2 - time_adapt_i_1

            preds_calib, preds_test = z[:len(labels_calib), :], z[len(labels_calib):, :]

            val_sets, time_fit_i, time_infer_i = conformal.conformal_method(
                args.ncscore, preds_calib, labels_calib, preds_test, args.alpha)

            metrics_conformal = evaluate_conformal(val_sets, labels_test, alpha=args.alpha)
            metrics_accuracy = accuracy(preds_test, labels_test, (1, 5))
            emp_cov.append(metrics_conformal[0]), set_size.append(metrics_conformal[1])
            class_covgap.append(metrics_conformal[2])
            time_adapt.append(time_adapt_i), time_conf_fit.append(time_fit_i), time_conf_inf.append(time_infer_i)
            top1.append(metrics_accuracy[0].item()), top5.append(metrics_accuracy[1].item())

            # Lưu cho các biểu đồ
            coverage_list.append({
                "Coverage": metrics_conformal[0],
                "Set Size": metrics_conformal[1],
                "Density": metrics_conformal[0],   # giả sử density = coverage, sửa lại nếu có cột khác
                "Method": args.adapt
            })
            time_list.append({
                "Method": args.adapt,
                "Seconds": time_adapt_i + time_fit_i + time_infer_i,
                "Component": "Total"
            })
            avg_size_list.append({
                "Temperature": args.epsilon,
                "Average Size": metrics_conformal[1],
                "Method": args.adapt
            })
            delta_list.append({
                "Delta Accuracy": metrics_accuracy[0].item() - metrics_accuracy[1].item(),
                "Delta Set Size": metrics_conformal[1],
                "Temperature": args.epsilon,
                "Accuracy": metrics_accuracy[0].item(),
                "Set Size": metrics_conformal[1]
            })

        results_detailed[args.test_datasets[i_domain]] = {
            "cov": emp_cov,
            "set_size": set_size,
            "class_covgap": class_covgap,
            "top1": top1
        }

        res_i = {"backbone": args.backbone, "dataset": args.test_datasets[i_domain], "alpha": args.alpha,
                 "adapt": args.adapt, "ncscore": args.ncscore, "epsilon": args.epsilon,
                 "ot_iters": args.ot_iters, "observed_marginal": str(args.observed_marginal),
                 "prop. calib": args.p, "top1": np.round(np.median(top1), 3), "cov":  np.round(np.median(emp_cov), 3),
                 "size": np.round(np.median(set_size), 2), "CCV":  np.round(np.median(class_covgap), 3),
                 "time_adapt": np.round(np.mean(time_adapt), 6), "time_conf_fit": np.round(np.mean(time_conf_fit), 6),
                 "time_conf_inf": np.round(np.mean(time_conf_inf), 6)}
        res = pd.concat([res, pd.DataFrame(res_i, index=[0])])

    # Produce average results
    avg = res[["top1", "cov", "size", "CCV", "time_adapt", "time_conf_fit", "time_conf_inf"]].mean().values
    res_avg = {"backbone": args.backbone, "dataset": "AVG", "alpha": args.alpha, "adapt": args.adapt,
               "ot_iters": args.ot_iters, "observed_marginal": str(args.observed_marginal),
               "ncscore": args.ncscore, "epsilon": args.epsilon, "prop. calib": args.p,
               "top1": np.round(avg[0], 3), "cov": np.round(avg[1], 3), "size": np.round(avg[2], 2),
               "CCV": np.round(avg[3], 3), "time_adapt": np.round(avg[4], 6), "time_conf_fit": np.round(avg[5], 6),
               "time_conf_inf": np.round(avg[6], 6)}
    res = pd.concat([res, pd.DataFrame(res_avg, index=[0])])

    timestamp = datetime.now().strftime("-%m-%d_%H-%M-%S")
    path = "./local_data/results/{backbone}/{alpha}/{ncscore}/summary/".format(
        backbone=args.backbone.replace("/", ""), alpha=str(args.alpha).replace(".", ""),
        ncscore=args.ncscore)
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame.to_excel(res, path + args.adapt + timestamp + ".xlsx")

    path = "./local_data/results/{backbone}/{alpha}/{ncscore}/detailed/".format(
        backbone=args.backbone.replace("/", ""), alpha=str(args.alpha).replace(".", ""),
        ncscore=args.ncscore)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + args.adapt + timestamp + ".npy", results_detailed)

    # ==== VẼ BỐN BIỂU ĐỒ & LƯU PKL ====
    coverage_df = pd.DataFrame(coverage_list)
    time_df = pd.DataFrame(time_list)
    avg_size_df = pd.DataFrame(avg_size_list)
    delta_df = pd.DataFrame(delta_list)

    # 1. Coverage vs Set Size
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Coverage", y="Set Size", data=coverage_df)
    plt.title("Empirical Coverage vs Set Size")
    plt.savefig("coverage_vs_set_size.png")
    plt.show()

    # 2. Coverage vs Density (ảnh 3)
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="Coverage", y="Density", hue="Method", data=coverage_df)
    plt.title("Empirical Coverage vs Density")
    plt.savefig("coverage_vs_density.png")
    plt.show()

    # 3. Time Comparison (ảnh 4)
    plt.figure(figsize=(7, 5))
    sns.barplot(x="Method", y="Seconds", hue="Component", data=time_df)
    plt.title("Time Comparison")
    plt.savefig("time_comparison.png")
    plt.show()

    # 4. Average Size vs Temperature (ảnh 5)
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="Temperature", y="Average Size", hue="Method", data=avg_size_df)
    plt.title("Average Size vs Temperature")
    plt.savefig("avg_size_vs_temp.png")
    plt.show()

    # 5. Delta Set Size vs Delta Accuracy (ảnh 6a)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x="Delta Accuracy", y="Delta Set Size", data=delta_df)
    plt.title("Delta Set Size vs Delta Accuracy")
    plt.savefig("delta_setsize_vs_delta_acc.png")
    plt.show()

    # 6. Accuracy và Set Size theo Temperature (ảnh 6b)
    plt.figure(figsize=(7, 5))
    sns.lineplot(x="Temperature", y="Accuracy", data=delta_df, label="Accuracy")
    sns.lineplot(x="Temperature", y="Set Size", data=delta_df, label="Set Size")
    plt.title("Accuracy and Set Size vs Temperature")
    plt.legend()
    plt.savefig("acc_setsize_vs_temp.png")
    plt.show()

    # Lưu các DataFrame ra file pkl
    with open('coverage_df.pkl', 'wb') as f:
        pickle.dump(coverage_df, f)
    with open('time_df.pkl', 'wb') as f:
        pickle.dump(time_df, f)
    with open('avg_size_df.pkl', 'wb') as f:
        pickle.dump(avg_size_df, f)
    with open('delta_df.pkl', 'wb') as f:
        pickle.dump(delta_df, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_datasets',
                        default='dtd,aircraft',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--backbone', default='CLIP-ViT-B/16')
    parser.add_argument('--adapt', default='linear_probe', choices=['none', 'linear_probe', 'confot', 'tim', 'transclip'])
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--ot_iters', default=3, type=int)
    parser.add_argument('--observed_marginal', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--ncscore', default='lac', choices=['lac', 'aps', 'raps'])
    parser.add_argument('--p', default=0.5, type=float)
    parser.add_argument('--seeds', default=20, type=int)
    args, unknown = parser.parse_known_args()
    process(args=args)

if __name__ == "__main__":
    main()
