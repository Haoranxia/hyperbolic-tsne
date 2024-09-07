# What do I want to investigate and achieve?
What does the big picture of this thesis look lke? What does it look like right now? What are some promising directions? 

Main purpose is to analyze the behaviour of this method (hyperbolic t-sne with accel.). I guess there's 2 parts to the thesis:
1. Investigations into the accelerated embedding specifically
    - Is there anything I can add/change/optimize/improve regarding the acceleration structure?
    Not really I think... (TBD)
    This part can be put on the backburner for a bit
      

2. Investigations into correct vs incorrect gradient? [BIG THEME]
    - Are there any differences in embedding quality, convergence, etc..?
    - Any qualitative predictions? Hypotheses? 
    - How do things look like if implement alternative cost functions?
      This one will require some work. I might have to derive the gradients analytically and then implement them.
      Alternatively, I could come up with some autograd version that still incorporates the acceleration scheme.
    - Ideally have a baseline dataset (hyperbolic/hierarchical dataset)


3. Investigations into hyperbolic embeddings/visualizations generally?
    - For example, how does our method compare to Poincare Embeddings?
    - Ideally have a guaranteed hyperbolic/hierarchical dataset (WordNet? Generate one such dataset?)
      This can be used as a basic dataset to experiment with (different losses, parameters, etc...).
      Also serves as a good sanity check.

### Notes:
1. Full WordNet data with acceleration converges very fast. (Without accel. very slow)

2. We can't apply autodiff to the gradients since it neglects simplifications of terms in the gradient
   which can potentially be cancelled out (for example in the t-sne gradient). Autodiff will compute the full terms and thus cause overhead. Therefore the only way to get an "efficient" gradient is by deriving it analtically and simplifying it manually.

3. Gradients on a spectrum; The 3 current gradients (wrong one, correct one, 3rd one?; or the 3 gradients from h-sne, co-sne, poincare maps?) could potentially be analyzed through the attraction-repulsion spectrum idea

4. Oct-tree in lorentz model and compare performance as a direction?

5. Neighbourhood embedding methods. We assume there is a probability distribution over every data point (so that we capture the neighbourhood of a point) and try to model that distribution in our embedding too. (Hence neighbourhood embedding). 

6. Co-SNE gradient conceptually does not make a lot of sense. Since it takes the difference in norms as an additional gradient term, but the norm of the embeddings 
||y|| is very small (less than 1), while ||x|| may be much bigger. (Due to difference in the nature of the spaces x, y live in). So I wonder how useful this gradient even is?

7. Global-hSNE gradient. We add an additional term corresponding to the global connection of the data. This term looks the same as the regular gradient, except that p_ij is now modelled using a t-distribution (as opposed to gaussian). However, we can't compute p_ij for every data point as we quickly run out of compute, resulting in having to use a (sparse) approximation. (i.e. use knn search to produce neighbours over which we compute t-distrib. distances over)

8. Is it possible to look into UMAP for hyperbolic visualizations? Does that provide a general approach for data originating from different kinds of spaces?

9. Why was the **grad_scale_fix** parameter set to **False**? Isn't this necessary for rescaling our gradient coming from Euclidean space to Hyperbolic space? What does it mean to not use **grad_scale_fix**?

10. How can we check quality of embeddings? (See paper evaluation section)
Right now we're only relying on visual judgement

### TODO Today ###
1. Run MNIST, C_elegans experiments.    X
   HyperbolicKL, BH approx, scale_fix=True, Correct gradient and incorrect gradient
   - Make sure correct folders are set up       X
   - Make sure file names are correct           X
2. Finish implementing global hsne functions (Implement while running experiments)  X
3. Test global hsne and cosne gradient
4. Read evaluation metric section of paper

### TODO Later ###
1. Run experiments on PC (probably has higher processing power, lets try it)
2. Visualize gradients
3. Experiment with smaller data set, with gradient field visualization, and without approximation to see differences in gradients
4. Compare gradients of fixed and old gradient
  - mnist sample 2 from each cluster (20 points)
  - track gradients of every step for both versions
  - I can store the gradients at every iteration step, (i.e. list of gradient matrices?)
  - Then somehow visualize these as vectors [Requires some researching of visualization algorithms]
  - visualize?
  - play around with position of clusters and how that affects the gradient
5. For CoSNE, embedding norms, use hyperbolic metric
6. Poincare maps gradient derivation = another contribution, derive it
7. global hsne gradient can be accelerated since the gradient simplifies. Try this too

### Notes on gradients comparing ###
- I can't just sample 10 points and then perform sgd. This is because that would not be enough points for the algorithm to run, i.e. the program just converges because the low dimensional embeddings are already good enough since 10 points are easy to embed. 

- Maybe I can try sampling like 1000 points but tracking just a few of them to see what is going on?

To have an effective loss, we need a decently sized high dimensional embedding. Otherwise the gradient would already be in a optima since there's a high chance our initial embedding is already good enough.

Another way to tackle this, have only 2 classes, sample 10 points from each, and then randomly initialize, and track iterations. This forces the idea that our initial embedding probably doesn't reflect the higher dimensional one, thus we will converge to something.


- Think about a proper experimen

