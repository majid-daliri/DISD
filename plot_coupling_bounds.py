import numpy as np
import argparse
import pandas as pd
import torch

import matplotlib
from matplotlib import pyplot as plt


def sanitize_filename(text):
    sanitized = text.replace(" ", "_").replace("?", "")
    return sanitized


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare coupling methods for model distributions and plot results including bounds.")
    parser.add_argument("--model_large", type=str, required=True, help="Name of the large model")
    parser.add_argument("--model_small", type=str, required=True, help="Name of the small model")
    parser.add_argument("--input_text", type=str, required=True,
                        help="Input text used for generating the token distributions")
    parser.add_argument("--num_tokens", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--num_retries", type=int, default=20000, help="Number of retries for statistical tests")
    return parser.parse_args()


def d_tv(p, q):
    return 1 - torch.sum(torch.min(p, q))


def gumbel_simulation(p, q, seed=42):
    torch.manual_seed(seed)
    num_elements = len(p)
    h = torch.rand(num_elements).to('cuda')

    choice_p = torch.argmin(-torch.log(h) / p)
    choice_q = torch.argmin(-torch.log(h) / q)

    return choice_p == choice_q


def wmh_simulation(p, q, seed=42):
    torch.manual_seed(seed)

    max_vals = torch.max(p, q)
    sum_max_vals = torch.sum(max_vals).item()

    i = j = None
    while i is None or j is None:
        dart = torch.rand(1).item() * sum_max_vals

        cumulative_sum = torch.cumsum(max_vals, dim=0)

        hit_index = torch.searchsorted(cumulative_sum, torch.tensor(dart, device='cuda')).item()
        hit_point = (dart - (cumulative_sum[hit_index] - max_vals[hit_index]).item())
        if i is None and hit_point <= p[hit_index]:
            i = hit_index

        if j is None and hit_point <= q[hit_index]:
            j = hit_index
    return i == j


def calculate_coupling_bounds(args):
    df_large = pd.read_csv(f"./data/{sanitize_filename(args.input_text)}/{args.model_large}_probs.csv")
    df_small = pd.read_csv(f"./data/{sanitize_filename(args.input_text)}/{args.model_small}_probs.csv")

    results = []
    for i in range(args.num_tokens):
        p = torch.tensor(df_large.loc[i].to_numpy(), dtype=torch.float64).to('cuda')
        q = torch.tensor(df_small.loc[i].to_numpy(), dtype=torch.float64).to('cuda')

        res_gumbel = sum(gumbel_simulation(p, q, seed=args.num_retries * i + ret) for ret in
                         range(args.num_retries)) / args.num_retries
        res_wmh = sum([wmh_simulation(p, q, seed=args.num_retries * i + ret) for ret in
                       range(args.num_retries)]) / args.num_retries
        dtv = d_tv(p, q)

        sum_max = torch.sum(torch.max(p, q))
        min_pq = torch.min(p, q)
        max_pq = torch.max(p, q)

        tlw = torch.sum(min_pq / (min_pq + sum_max - max_pq))

        results.append({
            'Token': i + 1,
            'DTV': dtv.item(),
            'Gumbel': res_gumbel,
            'WMH': res_wmh,
            'Tight Lower Bound': tlw.item(),
            'Upper Bound': 1 - dtv.item(),
            'Lower Bound': (1 - dtv.item()) / (1 + dtv.item())
        })

    results_df = pd.DataFrame(results)
    return results_df


def plot_coupling_bounds(args, results_df):
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    matplotlib.rc('font', family='serif', size=50)

    x = np.linspace(0, 1, 1000)

    y1 = 1 - x
    y2 = (1 - x) / (1 + x)

    plt.figure(figsize=(18, 14))
    line1, = plt.plot(x, y1,
                      label=r'$1-D_{\mathrm{TV}}(\mathcal{P},\mathcal{Q})$ \fontfamily{cmr}\selectfont{(Optimal)}',
                      color="#1f77b4", linewidth=7)
    line2, = plt.plot(x, y2,
                      label=r'$\frac{1-D_{\mathrm{TV}}(\mathcal{P},\mathcal{Q})}{1+D_{\mathrm{TV}}(\mathcal{P},\mathcal{Q})}$ \fontfamily{cmr}\selectfont{(Thm. 3 Bound)}',
                      color="#d62728", linewidth=7)

    first_legend = plt.legend(handles=[line1, line2], loc='best', fontsize=52)

    plt.gca().add_artist(first_legend)

    scatter1 = plt.scatter(results_df['DTV'], results_df['Gumbel'], color="#8B008B", edgecolors='none',
                           label=r'\fontfamily{cmr}\selectfont{Gumbel Sampling}', alpha=1, marker='o', s=500)
    scatter2 = plt.scatter(results_df['DTV'], results_df['WMH'], color="#556B2F", edgecolors='none',
                           label=r'\fontfamily{cmr}\selectfont{Weighted MinHash}', alpha=1, marker='D', s=350)

    plt.legend(handles=[scatter1, scatter2], loc='lower left', fontsize=52)

    plt.xlabel(r'$D_{\mathrm{TV}}(\mathcal{P},\mathcal{Q})$', fontsize=52)
    plt.ylabel(r'$\Pr[a = b]$', fontsize=52)
    plt.title(r"\fontfamily{cmr}\selectfont{" + f"{args.input_text}" + "}", fontsize=52)

    plt.grid(True)
    plt.savefig(f"{sanitize_filename(args.input_text)}.png", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    results_df = calculate_coupling_bounds(args)
    plot_coupling_bounds(args, results_df)
