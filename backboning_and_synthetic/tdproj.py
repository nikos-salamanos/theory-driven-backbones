import torch, sys
import numpy as np
import pandas as pd
from sbi.inference import NPE
from sbi.utils import BoxUniform

PRIOR_LOW = [0.0, 0.0, 0.0]
PRIOR_HIGH = [3.0, 3.0, 4.0]

def _random(theta, dist, sizes1, sizes2):
   w = np.random.uniform(size = (theta.shape[0], sizes1.shape[0], sizes2.shape[0]))
   return w / w.sum(axis = (1, 2), keepdims = True)

# This function implements the actual gravity model, with theta parameters, and distance matrix and size distribution for the agents
def _gravity(theta, dist, sizes1, sizes2):
   # The baseline gravity model formula:
   # theta[0] is scaling for the sizes of the first type of agent
   # theta[1] is scaling for the sizes of the first type of agent
   # theta[2] is the scaling for the distance measure
   # All variables are logtransformed
   logs = (np.log(sizes1)[:, None] * theta[:, 0, None, None]
         + np.log(sizes2)[None, :] * theta[:, 1, None, None]
         - np.log(dist)[None] * theta[:, 2, None, None])
   logs -= logs.reshape(len(theta), -1).max(axis = 1)[:, None, None]
   # Undo the log transformation
   w = np.exp(logs)
   # Return normalized bipartite networks
   return w / w.sum(axis = (1, 2), keepdims = True)

MODELS = {"gravity": _gravity, "random": _random}

# Given a bipartite weighted adjacency matrix w, the function estimates the parameters of the gravity model
# First the gravity mass of the first type of node, then of the second, and finally the importance of the distance
def _summaries(w, dist, sizes1, sizes2):
   return np.stack([(w * np.log(sizes1)[:, None]).sum(axis = (1, 2)),
                    (w * np.log(sizes2)[None, :]).sum(axis = (1, 2)),
                    (w * np.log(dist)[None]).sum(axis = (1, 2))], axis = 1)

# This function creates and returns a wrapper function for the ABM model so that it can be used by NPE
def _simulator(dist, sizes1, sizes2, model):
   model_f = MODELS[model]
   def sim(theta):
      # Get the prior parameters
      t = theta.detach().cpu().numpy() if torch.is_tensor(theta) else np.asarray(theta)
      # Run the ABM with the parameters
      w = model_f(t, dist, sizes1, sizes2)
      # Return the posteriors from the simulated bipartite w 
      return torch.tensor(_summaries(w, dist, sizes1, sizes2), dtype = torch.float32)
   return sim

# This function fits the ABM on the observed data
# X, dist, sizes1, sizes2 are what is needed for the fit (observed data, distance matrix, sizes of the nodes of both types
# We do n_sims simulation and extract n_post posteriors
def _calibrate(X, dist, sizes1, sizes2, model = "gravity", n_sims = 5000, n_post = 5000, seed = 0):
   torch.manual_seed(seed)
   # We create a set of priors chosen uniformly in an n-dimensional space between PRIOR_LOW and PRIOR_HIGH (n dependent on the ABM's number of parameters)
   prior = BoxUniform(low = torch.tensor(PRIOR_LOW), high = torch.tensor(PRIOR_HIGH))
   sys.stderr.write("   Running simulator...\n")
   # Get the simulation wrapper function
   sim = _simulator(dist, sizes1, sizes2, model)
   # Sample parameter combinations from the prior
   theta = prior.sample((n_sims,))
   sys.stderr.write("   Making inference...\n")
   # Set up the inference machine
   inference = NPE(prior = prior).append_simulations(theta, sim(theta))
   # Train the neural network and get the posterior from it
   posterior = inference.build_posterior(inference.train())
   # Get the summary parameter values from the observed data
   x_obs = torch.tensor(_summaries((X / X.sum())[None], dist, sizes1, sizes2)[0], dtype = torch.float32)
   # Sample posterior parameters
   return posterior.sample((n_post,), x = x_obs).detach().cpu().numpy()

# This function performs a resampling with the w extracted bipartite matrices from the ABM
# It needs to know the total edge weight sum from the observed bipartite network ("total").
def _resample(w, total, rng):
   # We flatten out the k sampled bipartite realizations from the ABM so that each row of "flat" has the weights of each bipartite edge
   flat = w.reshape(len(w), -1)
   # Normalize each bipartite network weight row so that it sums to one (stochastic)
   flat = flat / flat.sum(axis = 1, keepdims = True)
   # We now draw "total" number of edge weights, so that the total edge weight sum of each sampled bipartite network is equal to the one of the observed one
   drawn = rng.multinomial(int(total), flat).astype(float)
   # We reconstruct the 3 dimensional tensor with k generated bipartite matrices stochastic adjacency matrices
   return (drawn / np.maximum(drawn.sum(axis = 1, keepdims = True), 1)).reshape(w.shape)

# This function is sampling n_draws realizations of the bipartite structure from the fitted ABM
# Differently from _accumulate below, it does not compare them with the observed structure
# So it gives a bipartite networks tructure as understood by the ABM (the background model assumed by the theory)
# post has the posterior of the ABM, dist, sizes1, and sizes2 are the data the fitted ABM needs
# total is the sum of the bipartite's edge weights, obs are both projections of the observed bipartite structure
def _predictive(post, dist, sizes1, sizes2, total, n_draws, model, seed = 0):
   model_f = MODELS[model]
   rng = np.random.default_rng(seed)
   # We pick n_draws different parameter choices from the fitted ABM
   pick = rng.choice(len(post), size = n_draws, replace = n_draws > len(post))
   # We return the bipartite networks from realizations given by the ABM with the picked parameter choices
   return _resample(model_f(post[pick], dist, sizes1, sizes2), total, rng)

# This is the function that samples multiple different realizations of the bipartite structure from the fitted ABM
# post has the posterior of the ABM, dist, sizes1, and sizes2 are the data the fitted ABM needs
# total is the sum of the bipartite's edge weights, obs are both projections of the observed bipartite structure
# n_draws is how many realizations we'll make, measure is what measure we use for the projection of the generated bipartites
def _accumulate(post, dist, sizes1, sizes2, total, obs, E, n_draws, measure, model, seed, chunk = 200):
   rng = np.random.default_rng(seed)
   model_f = MODELS[model]
   # We create zero-filled data structures for each projection that will be updated at each simulation
   acc = [{k: np.zeros_like(o) for k in ("s", "ss", "le")} for o in obs]
   done = 0
   while done < n_draws:
      sys.stderr.write(f"   Draw {done}...\r")
      # We pick k different parameter choices from the fitted ABM ("chunk" or fewer if we're almost done)
      k = min(chunk, n_draws - done)
      pick = rng.choice(len(post), size = k, replace = k > len(post) or n_draws > len(post))
      # We now sample k bipartite networks from realizations given by the ABM with the picked parameter choices
      w = _resample(model_f(post[pick], dist, sizes1, sizes2), total, rng)
      # We update information for both projections
      for axis, o in enumerate(obs):
         # Project one way with a given similarity measure
         v = _project(_residuals(w, E), axis, measure)
         # We accumulate the total edge weight sum for each edge in the projection
         acc[axis]["s"] += v.sum(0)
         # We accumulate the total squared edge weight sum for each edge in the projection (for the standard deviation)
         acc[axis]["ss"] += (v ** 2).sum(0)
         # We count how many times the generated edge weight is lower or equal to the observed edge weight
         acc[axis]["le"] += (v <= o).sum(0)
      done += k
   sys.stderr.write(f"\n")
   # We keep track of how many draws we made
   for a in acc:
      a["n"] = n_draws
   return acc

# The fitted model's expected bipartite matrix, averaged over the posterior (in chunks to bound memory)
def _expected(post, dist, sizes1, sizes2, model, chunk = 500):
   parts = np.array_split(post, max(1, len(post) // chunk))
   return np.mean([MODELS[model](p, dist, sizes1, sizes2).mean(axis = 0) for p in parts], axis = 0)

# Pearson residuals: what is left in each bipartite cell once the model's expectation is removed
# The model is multiplicative in the cells, so this is where it has to be removed, before projecting
def _residuals(w, E):
   return (w - E) / np.sqrt(np.maximum(E, 1e-300))

# This makes a simple projection from a bipartite structure w
def _project(w, axis = 0, measure = "cooccurrence"):
   # Convenience line depending on how we got here (from simulation or original data)
   m = (w if w.ndim == 3 else w[None])
   # On which axis we need to project
   m = m if axis == 0 else m.transpose(0, 2, 1)
   # We pick the similarity measure for the projection, which requires a different normalization
   if measure == "cooccurrence":
      m = m - m.mean(axis = 2, keepdims = True)
   if measure in ("cosine", "correlation"):
      m = m / np.maximum(np.linalg.norm(m, axis = 2, keepdims = True), 1e-300)
   # Perform the projection
   s = np.einsum("sij,skj->sik", m, m)
   idx = np.arange(s.shape[1])
   s[:, idx, idx] = 0
   return s

# Assign the weights to the edges
def _weights(s, labels):
   i, j = np.triu_indices(len(labels), k = 1)
   return pd.DataFrame({"src": np.asarray(labels)[i], "trg": np.asarray(labels)[j], "nij": s[i, j]}).sort_values("nij", ascending = False).reset_index(drop = True)

# This function takes the baseline model expectations (from parameter a) and compares it with the observed structure
# The result is a score and p_value that gives us an estimation of whether the edge exists or not
def _edgelist(a, observed, labels):
   # We only look at the upper triangular part, because it is an undirected network
   i, j = np.triu_indices(len(labels), k = 1)
   # We reconstruct the expected edge weight from the ABM simulations
   n = a["n"]
   expected = a["s"][i, j] / n
   # This is the standard deviation of the expectation
   sd = np.sqrt(np.maximum(a["ss"][i, j] / n - expected ** 2, 0))
   # probability that the observed edge is above the expectation from the ABM
   p_low = (1 + a["le"][i, j]) / (n + 1)
   # Put all together in a dataframe
   return pd.DataFrame({
      "src": np.asarray(labels)[i],
      "trg": np.asarray(labels)[j],
      "observed": observed[i, j],
      "expected": expected,
      "score": np.divide(observed[i, j] - expected, sd, out = np.zeros(len(i)), where = sd > 0),
      "p_low": p_low,
   }).sort_values(by = "score", ascending = False).reset_index(drop = True)

# Convenience function to move from node ids to labels
def _labelled(X, labels1, labels2):
   X = np.asarray(X, dtype = float)
   l1 = np.arange(X.shape[0]) if labels1 is None else labels1
   l2 = np.arange(X.shape[1]) if labels2 is None else labels2
   return X, l1, l2

# This is the function that produces the theory-driven backbone, that is the edges that overcome the baseline information captured by the ABM
def backbone(X, dist, sizes1, sizes2, labels1 = None, labels2 = None, measure = "cooccurrence", model = "gravity", n_draws = 5000, seed = 0, chunk = 200, post = None, **kwargs):
   # Making sure we know which node id corresponds to which node label
   X, l1, l2 = _labelled(X, labels1, labels2)
   sys.stderr.write("Calibrating model...\n")
   # We need the posterior from the fitted ABM to test against
   post = _calibrate(X, dist, sizes1, sizes2, model = model, seed = seed, **kwargs) if post is None else post
   # We project the observed bipartite into its two projections
   # We can choose which bipartite projection technique we want to use, for the test case cosine similarity is best
   # The theory is removed at the level of the bipartite cells, then what is left is projected
   E = _expected(post, dist, sizes1, sizes2, model)
   obs = [_project(_residuals(X / X.sum(), E), axis, measure)[0] for axis in (0, 1)]
   sys.stderr.write("Making null model...\n")
   # We need to run the fitted ABM a sufficient number of times to get enough information to model each edge
   acc = _accumulate(post, dist, sizes1, sizes2, X.sum(), obs, E, n_draws, measure, model, seed, chunk)
   sys.stderr.write("Perform projection...\n")
   # Then we can produce the final backbone by looking at which observed edges surpass the expectation we just built
   return (_edgelist(acc[0], obs[0], l1), _edgelist(acc[1], obs[1], l2))

# This is the function that produces a network as described by the fitted ABM, so without comparing with the actual observed one
def theory(X, dist, sizes1, sizes2, labels1 = None, labels2 = None, measure = "cooccurrence", model = "gravity", n_draws = 2000, seed = 0, post = None, **kwargs):
   # Making sure we know which node id corresponds to which node label
   X, l1, l2 = _labelled(X, labels1, labels2)
   # We need the posterior from the fitted ABM to generate the projection
   post = _calibrate(X, dist, sizes1, sizes2, model = model, seed = seed, **kwargs) if post is None else post
   # We now use the fitted ABM to simulate the structure many times to get a proper estimation of whether the edge should be there or not for the model
   null = _predictive(post, dist, sizes1, sizes2, X.sum(), n_draws, model, seed)
   # Once we have created enough simulated bipartites, we can project them using a given bipartite projection technique
   return (_weights(_project(null, 0, measure).mean(axis = 0), l1), _weights(_project(null, 1, measure).mean(axis = 0), l2))

# This is just a convenience function producing the projection without backboning
# It is the backbone one would get without doing any backboning step
def empirical(X, labels1 = None, labels2 = None, measure = "cooccurrence"):
   # Making sure we know which node id corresponds to which node label
   X, l1, l2 = _labelled(X, labels1, labels2)
   # We don't need to do anything for the projection, just normalize the bipartite
   obs = X / X.sum()
   # And then project it
   return (_weights(_project(obs, 0, measure)[0], l1), _weights(_project(obs, 1, measure)[0], l2))
