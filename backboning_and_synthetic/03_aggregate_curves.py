import numpy as np
import pandas as pd
from sklearn.metrics import auc

for combo in (("pr_curves", "recall"), ("roc_curves", "fpr")):
   df = pd.read_csv(f"{combo[0]}.tsv", sep = "\t")
   df = df.groupby(by = ["tag", "method", combo[1]]).agg(["mean", "count", "std"])
   df.columns = ("avg", "count", "std")
   df["ci95"] = (1.96 * df["std"]) / np.sqrt(df["count"])
   df = df.reset_index()
   df[df["tag"] == "A"].pivot_table(index = combo[1], columns = "method", values = ["avg", "ci95"]).reset_index().to_csv(f"{combo[0].split('_')[0]}_A.tsv", index = False, sep = "\t")
   df[df["tag"] == "S"].pivot_table(index = combo[1], columns = "method", values = ["avg", "ci95"]).reset_index().to_csv(f"{combo[0].split('_')[0]}_S.tsv", index = False, sep = "\t")
   for tag in ["A", "S"]:
      for method in ["naive", "df", "nc", "random", "gravity"]:
         cell = df[(df["tag"] == tag) & (df["method"] == method)]
         auc_mean = auc(cell[combo[1]], cell["avg"])
         auc_low = auc_mean - auc(cell[combo[1]], cell["avg"] - cell["ci95"])
         print_curve = f"\\multirow{{10}}{{*}}{{PR}}" if method == "naive" and combo[0] == "pr_curves" and tag == 'A' else f"\\multirow{{10}}{{*}}{{ROC}}" if method == "naive" and combo[0] == "roc_curves" and tag == 'A' else ""
         print_tag = f"\\multirow{{5}}{{*}}{{{tag}}}" if method == "naive" else ""
         if method == "gravity":
            print(f"{print_curve} & {print_tag} & {method} & $\\textbf{{{auc_mean:.3f}}} \\pm {auc_low:.3f}$ \\\\")
         else:
            print(f"{print_curve} & {print_tag} & {method} & ${auc_mean:.3f} \\pm {auc_low:.3f}$ \\\\")
         if method == "gravity":
            if tag == "S" and combo[0] == "pr_curves":
               print("\\hline")
            else:
               print("\\cline{2-4}")

