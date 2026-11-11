import torch
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

# Template of synthetic generator:
# Step 1: Make "real" projections with two SBMs
# Step 2: Make a baseline gravity model
# Step 3: Update the baseline model with the real projection, increasing the p of S-A links if the s and a nodes are close in their projections
# Step 4: Sample the most likely discrete weighted bipartite network from the augmented model 

# We use ER to determine the oversampling for bipartite edges connecting nodes nearby in the real projections
def effective_resistance(projection):
   Linv = torch.linalg.pinv(torch.from_numpy(nx.laplacian_matrix(projection).todense().astype(float)), hermitian = True)
   pinv_diagonal = torch.diagonal(Linv)
   return pinv_diagonal.unsqueeze(0) +  pinv_diagonal.unsqueeze(1) - 2 * Linv

# This function makes the "real" networks that should be recovered by the backboning from the framework
# We'll use an SBM with n_groups communities, whose size scales with a given exponent
# Scale controls how many nodes, so we can make one projection smaller than the other
# mu and avg_comm_degree tells us how well defined communities are (edges out vs edges in)
# tag is S for sites and A for amphoras
def make_projection(n_groups, exponent, scale, mu, avg_comm_degree, tag):
   projection = nx.Graph()
   # Because I want to use effective resistance, I want a single connected component
   # I also want a minimum number of nodes to get enouhc signal
   while nx.number_connected_components(projection) != 1 or len(projection.nodes) < 30:
      # First we decide the size of the communities in the SBM, with a power law
      commsizes = np.ceil((1 - np.random.power(exponent, size = n_groups)) * scale).astype(int)
      # We cannot have communities with a single node in it (otherwise the p_in is undefined)
      while commsizes.min() < 2:
         commsizes = np.ceil((1 - np.random.power(exponent, size = n_groups)) * scale).astype(int)
      # All communities pairs have the same low probability of edges between them (regulated by mu)
      probs = np.full((n_groups, n_groups), mu / commsizes.sum())
      # Each community has an internal p_in connection probability proportional to avg_comm_degree
      for i in range(len(commsizes)):
         probs[i,i] = min(2 * avg_comm_degree / (commsizes[i] - 1), 1)
      # Wire up the network
      projection = nx.stochastic_block_model(commsizes, probs)
   # We need to make sure to have a canonical edge form (lower id first) for undirected networks
   df = nx.to_pandas_edgelist(projection)
   df.columns = ("src", "trg")
   df["src"], df["trg"] = df[["src", "trg"]].min(axis = 1), df[["src", "trg"]].max(axis = 1)
   df["src"], df["trg"] = df["src"].map(lambda x: f"{tag}{x}"), df["trg"].map(lambda x: f"{tag}{x}")
   df.to_csv(f"sbm_proj_sample_{tag}.txt", sep = "\t", index = False)
   return projection

# This function makes the baseline background model the ABM will then try to infer
def base_gravity(proj1, proj2):
   # The travel cost between each pair of sites and amphoras is modeled to be random with a hard bottom
   travel_cost = np.random.normal(size = (len(proj1.nodes), len(proj2.nodes)))
   travel_cost += np.abs(travel_cost.min()) + 0.5
   np.savetxt("dists.csv", travel_cost, fmt = "%.8f", delimiter = "\t")
   # The sizes of settlements (both consuming an producing sites) is modeled via a lognormal
   sizes1 = np.random.lognormal(size = len(proj1.nodes))
   sizes2 = np.random.lognormal(size = len(proj2.nodes))
   np.savetxt("sizes_S.csv", sizes1, fmt = "%.8f", delimiter = "\t")
   np.savetxt("sizes_A.csv", sizes2, fmt = "%.8f", delimiter = "\t")
   # Trivial gravity model, each site pair connects proportionally to the two sizes and inversely to their distances
   base_model = np.outer(sizes1, sizes2) / travel_cost
   return sizes1, sizes2, base_model

# Here we use the network topology of the real projection to create a similarity matrix we can use to augment bipartite edge weights
def backbone_similarity(projection, temp = 0.1):
   # First we get the distance between all pairs of nodes in the projection
   R = effective_resistance(projection)
   # We normalize by the degree, otherwise the structural signal is too squished
   deg = torch.tensor([d for _, d in projection.degree()], dtype = torch.float64)
   R = torch.clamp(R - (1 / deg.unsqueeze(0) + 1 / deg.unsqueeze(1)), min = 0)
   # Then we transform distances into similarities, with a discount, then normalize so the size of the network does not induce the result
   sym = torch.exp(-R / temp)
   return (sym / sym.sum(dim = 1, keepdim = True)).numpy()

# A smoothing function not to dilute the structural signal from the projections too much
def smooth_latent(sym, rank):
   f = sym @ np.random.normal(size = (len(sym), rank))
   return (f - f.mean(axis = 0)) / f.std(axis = 0)

# Here we bring together the baseline gravity model and the 
def apply_backbones(base_model, sym1, sym2, rank = 2, strength = 2.0):
   # First we smooth the information from both projections
   F = smooth_latent(sym1, rank) @ smooth_latent(sym2, rank).T / np.sqrt(rank)
   # Then we apply them to the background gravity model
   model = base_model * np.exp(strength * F)
   return model / model.sum()

# Let's make the two projections: proj1 for sites, proj2 for amphoras
# Parameters are so that we always have a bit more sites, and the network topologies are slightly different
proj1 = make_projection(3, 1, 30, 0.8, 3, 'S')
proj2 = make_projection(4, 2, 20, 0.9, 3, 'A')

# Make the gravity model, we still need to know the sizes of the settlement to properly draw proportional to them
sizes1, sizes2, base_model = base_gravity(proj1, proj2)

# We calculate the effect of the real networks
sym1 = backbone_similarity(proj1)
sym2 = backbone_similarity(proj2)

# We now apply the projection information over the background gravity model
# Then we normalize to make it into a probability
true_model = apply_backbones(base_model, sym1, sym2)
observed_B = (true_model.T / true_model.sum(axis = 1)).T

# We must guarantee that the smallest site has at least one amphora and the smallest amphora is at least in one site
# If not, we multiply the weights by an increasing factor
while np.round(((sizes2 * observed_B).T * sizes1).T).astype(int).sum(axis = 1).min() < 1 or np.round(((sizes2 * observed_B).T * sizes1).T).astype(int).sum(axis = 0).min() < 1:
   observed_B *= 2

# We make this into an observed bipartite network with discrete edge weights
observed_B = np.round(((sizes2 * observed_B).T * sizes1).T).astype(int)

np.savetxt("observed_bipartite.csv", observed_B, fmt = "%d", delimiter = "\t")

print(observed_B.shape)
