# minibatch.size distributed uniformly between 10 and 100
# Smaller minibatches allow for more gradient updates, bigger ones are more
# precise.
# I think the 10-100 range comes from Hinton's "Practical Guide" to RBMs.
minibatch.size = sample(10:100, 1)

# n.ranef distributed uniformly
n.ranef = sample(5:25, 1)

# n.importance.samples distributed uniformly
# Tang and Salakhutdinov seemed to find the best performance with 20-30 samples.
n.importance.samples = sample(20:50, 1)

# n.layer2 uniformly distributed
# This layer does dimensionality reduction. With better priors, fewer dimensions
# would probably be needed, since there still tend to be strong correlations.
n.layer2 = sample(5:20, 1)

# n.layer1 uniformly distributed
# This layer does basis expansion, so it should be at least as wide as layer2,
# which does dimensionality reduction
n.layer1 = sample(n.layer2:50, 1)

# Starting learning rate currently fixed.
# Found this through trial and error.  Haven't messed with it much, since I 
# don't want any exploding gradients and I already have two other ways to
# trade off speed versus accuracy during gradient descent (# of samples and 
# minibatch size).
starting.rate = 0.05
