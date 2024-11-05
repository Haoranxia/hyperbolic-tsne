# Experiment notes

### 1
Today I experimented a bunch with GaussianKL, HyperbolicKL, with/without BH approx. using the custom tree dataset and noticed the following:
- HyperbolicKL produces bad visualizations. 
- GaussianKL seems to produce nice visualizations that retain the tree structure for our custom (tree-like) dataset.
- GaussianKL (for tree dataset) actually converges very quickly
- GaussianKL has a weird problem where the cost function becomes negative. However it dips to very small values (-0.08) which could be an numerical artifact due to small probabilities in the cost function. 
In the cost function when we take the log of possibly very small values. To avoid numerical issues, we take the max between that value and some eps. This may be the cause of these negative values.
- GaussianKL and HyperbolicKL with BH approx. sometimes gives extremely large gradients (with norms > e+29). This requires further investigation. 
- Early exaggeration may be the reason we get extremely large gradients
- Early exaggeration is very useful in obtaining better globally structured visualizations. 

