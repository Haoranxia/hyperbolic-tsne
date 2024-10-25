# Questions about Hyperbolic Space
This file contains my personal questions, uncertainties, etc.. about hyperbolic space, hyperbolic embeddings.



### 1: Interpreting Hyperbolic Embeddings (Poincare disk)
Hyperbolic embeddings are probably useful conceptually. When we embed data in hyperbolic space, the encoding may capture things such as hierarchical relationships. (Since hyperbolic space seems to capture this? I assume this is true)

However, once we visualize it (through a model, Poincare disk), how much of these "nice things" are retained?
**Q:**Are hierarchical structures possibly distorted? -- **A:**Definitely to some degree

The Poincare Disk only is able to show a tiny part of the whole (infinite) space. Most of the space is "squished" along the boundary, and thus most embeddings will end up there (probably).

This leads to not very "nice" or "useful" visualizations for a lot of general cases (I assume).

If we force data to be away from the boundary, more towards the center. So we force embeddings to occupy only a tiny amount of space. What happens to the visualization and embedding?
**Q:**Do we still have meaningful embeddings?
**Q:**What would a hierarchical relationship look like here? -- **A:**Intuitively it would be points "spreading" ou from the center

In the end, we want to be able to capture **hierarchical** relationships through our hyperbolic embedding visualizations. **Q:** How much of this is actually possible?



### 2: Hyperbolic probability distributions
We use t-sne in hyperbolic space to embed points. This means we define a hyperbolic probability distribution, and try to make the probabilities match the high dimensional one.

The naive idea is to just take any distribution, and replace the distance/metric function within with a hyperbolic equivalent one. **Q:** How "legal" is this? **A:** I suspect its alright because things are normalized nicely since we use some form of softmax probabilities. 



### 3: Understanding Hyperbolic Space
Perhaps I need to study differential geometrical concepts more in-depthly to gain an intuition for hyperbolic space? 
We can describe things like "more space", "exponential growth" using concepts like the metric tensor to convert between familiar Euclidean concepts and Hyperbolic spatial ones.

Maybe this also gives a more quantitative description of "distortion" of embeddings (visually) once we use the Poincare Disk model. Maybe we can get some better idea also as how to interpret points embedded in the Poincare Disk. 





# My understanding of Hyperbolic Embeddings

### Chapter 1: "Hyperbolic" data and its properties
**assumptions**: Hyperbolic space CAN embed trees with arbitrarily low distortion. Euclidean space CANNOT

Assume we have some perfectly hierarchical (tree-like) data in N-dimensions.
Meaning: every node has M children. To embed this "properly", where we don't distort distances between children/parents, and nodes and other (unrelated) nodes, we need an exponential amount of euclidean dimensions.

Consider distances in a tree-like structure. Distances between children from very different branches is very large. Naive embeddings with a naive neighbourhood graph construction method might not be able to capture those distances accurately. 
It might consider 2 children which in actuality are far apart (due to being on different branches) close together because of their position in (Euclidean) space.
So in algorithms such as t-sne, when the affinity matrix is constructed, it might "falsely" assign high affinities to spatially close, but relationally distant points.

Meaning, in euclidean space, they are physically close (atleast according to the initial embedding), but in actuality, they are distant. 

Does this mean a first step (for hyperbolic t-sne), is to construct the high dimensional affinity matrix in such a way where it "encodes" the tree-like structure already?

So if we want to construct a high-dim. affinity matrix, by making sure it follows some "tree-like" structure, how can we go about this?

Even when we have hierarchical data, possibly organized in a tree structure, it is difficult to just use euclidean methods to capture that structure. 
Since if we naively use neighbouring methods, and create distance/affinity matrices off of that, then points that are spatially close, but in actuality far (because they live on different branches, but spatially they get embedded closely) do not get their hierarchical ordering reflected in the constructed affinity matrix. 

**Possible algorithm:**
1. Use k-means? to find clusters, and a "single" point per cluster that can represent it.
2. We assume these cluster centroids represent "nodes" of our tree and we can use this idea to construct a "tree", and eventually an affinity matrix based on tree-distances
3. So compute distances by "traveling" along the tree we construct (navigate to closest mean cluster and use this to reach our node), and the total distance would be the distance between those 2 nodes.

Then hyperbolic t-sne would just end up being a way to visualize the tree-structure of data.

Some problems: 
1. P: What if we have a loop, (ex. 3 clusters have eachother as closest neighbour)
   S: Don't allow visitations to already visited nodes?



### Chapter 2: Visualizing Hyperbolic data (Poincare Disk)
We can only choose 2 dimensions for visualization purposes. This means our choice of embedding model "must" be the Poincare Disk. 
Does this bring any problems? How "faithfully" are things represented?