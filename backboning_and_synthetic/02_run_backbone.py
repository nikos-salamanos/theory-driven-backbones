import os
import numpy as np
import tdproj as td
import pandas as pd
import backboning as bb
from collections import defaultdict
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def make_eval_curves(df, tag, method):
   precision, recall, thresholds = precision_recall_curve(df["_merge"] != "left_only", df["score"])
   pr = pd.DataFrame()
   pr["recall"] = np.linspace(0, 1, 100)
   pr["precision"] = np.interp(np.linspace(0, 1, 100), recall[::-1], precision[::-1])
   pr["tag"] = tag
   pr["method"] = method
   pr.to_csv("pr_curves.tsv", index = False, header = not os.path.exists("pr_curves.tsv"), sep = "\t", mode = 'a')
   fpr, tpr, thresholds = roc_curve(df["_merge"] != "left_only", df["score"])
   roc = pd.DataFrame()
   roc["fpr"] = np.linspace(0, 1, 100)
   roc["tpr"] = np.interp(np.linspace(0, 1, 100), fpr, tpr)
   roc["tag"] = tag
   roc["method"] = method
   roc.to_csv("roc_curves.tsv", index = False, header = not os.path.exists("roc_curves.tsv"), sep = "\t", mode = 'a')

def evaluate_backbones(tag_method_infer):
   for tag in tag_method_infer:
      # We load the real edges
      real = pd.read_csv(f"sbm_proj_sample_{tag}.txt", sep = "\t")
      for method in tag_method_infer[tag]:
         # By merging with indicator true we know that those rows only in infer are not part of the real projection
         df = tag_method_infer[tag][method].merge(real, on = ["src", "trg"], how = "outer", indicator = True)
         make_eval_curves(df, tag, method)

def classical_backboning(bb_alg):
   result = {}
   B = pd.read_csv("observed_bipartite.csv", sep = "\t", header = None)
   for tag in ('S', 'A'):
      # Depending on which projection we're looking at, we need to transpose
      if tag == 'S':
         infer = B.dot(B.T).unstack().reset_index().rename(columns = {"level_0": "src", "level_1": "trg", 0: "nij"})
      else:
         infer = B.T.dot(B).unstack().reset_index().rename(columns = {"level_0": "src", "level_1": "trg", 0: "nij"})
      # No self loops
      infer = infer[infer["src"] != infer["trg"]]
      # Each backboning method has its scoring logic
      if bb_alg == "nc":
         infer = bb.noise_corrected(infer, undirected = True)
         infer["score"] /= infer["sdev_cij"]
         infer["score"] = infer["score"].fillna(infer["score"].min())
      elif bb_alg == "df":
         infer = bb.disparity_filter(infer, undirected = True)
      elif bb_alg == "naive":
         infer["score"] = infer["nij"] / infer["nij"].max()
      # We make sure we can merge the networks properly with the real projections
      infer["src"], infer["trg"] = infer["src"].map(lambda x: f"{tag}{x}"), infer["trg"].map(lambda x: f"{tag}{x}")
      result[tag] = infer.copy()
   return result['S'], result['A']

# Load all data: bipartite edges, sizes and distances to calibrate the ABM
X = np.loadtxt("observed_bipartite.csv", dtype = int)
dist = np.loadtxt("dists.csv", dtype = float)
sizes1 = np.loadtxt("sizes_S.csv", dtype = float)
sizes2 = np.loadtxt("sizes_A.csv", dtype = float)

tag_method_infer = defaultdict(lambda : defaultdict(pd.DataFrame))

for model in ("gravity", "random"):
   tag_method_infer['S'][model], tag_method_infer['A'][model] = td.backbone(
      X,
      dist,
      sizes1,
      sizes2,
      model = model,
      measure = "cosine",
      labels1 = [f"S{i}" for i in range(sizes1.shape[0])],
      labels2 = [f"A{i}" for i in range(sizes2.shape[0])],
      n_draws = sizes1.shape[0] * sizes2.shape[0]
   )

for bb_alg in ("nc", "df", "naive"):
   tag_method_infer['S'][bb_alg], tag_method_infer['A'][bb_alg] = classical_backboning(bb_alg)

# EValuate with AUC of ROC and PR curves
evaluate_backbones(tag_method_infer)
