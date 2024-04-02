# # SciML SANUM2024
# # Lab 6: Neural Differential Equations and DiffEqFlux.jl


# In this final lab we look at combining differential equations and
# neural networks, with the goal of "learning" dynamics based on training data.
# That is, consider an ODE of the form
# $$
# u' = f(u) + g(u)
# $$
# where we know $f$ (or if we don't know anything, $f = 0$) but don't know
# $g$. We can approximate $g$ by a neural network, and then we want to choose
# the parameters to fit data.

# Here we look at some simple examples, but the same techniques have been used
# in clinical trial accelleration for vaccine development by Moderna,
# climate change modelling and COVID prediction, see the [SciML Schowcase](https://sciml.ai/showcase/).

# **Learning Outcomes**
# 1. Combining neural networks and differential equations.
# 2. Deducing dynamics by training a neural network.
# 3. Using multiple optimisers to get good approximations.
# 4. Neural ODEs as layers in a NN.
# 5. Number classification via a NN with a Neural ODE layer.

using Lux, DifferentialEquations, Optimization, OptimizationOptimisers, Plots, Zygote, SciMLSensitivity,
            ComponentArrays, Random, LinearAlgebra, Test, Statistics


# ## 6.1 Learning dynamics

# We begin with a very simple ODE:
# $$
# u' = u - α u^3
# $$
# where we know $f(u) = u$ but suppose we don't know $g(u) = -α u^2$.
# First let's setup some training data with different initial conditions.
# We will do 10 trials which are sampled at 15 points for $t ∈ [0,5]$.

##


# We will now try to deduce the term $-αu^3$ by training a simple NN
# by minimising the error when comparing the model to the provided data.
# Because Optimzation.jl (currently) requires that parameters behave like
# arrays, rather than passing in the NN as a parameter we make it
# a global constant. We begin with simple 2-layer piecewise affine NN:

##

# Our model is
# $$
#   u' = u + g(u)
# $$
# where we represent $g$ by a NN with given parameters. Here is the rhs for this simple model:

##

# We can then compute the loss by solving the ODE with a given set of parameters
# for each of the runs in our samples and summing over the 2-norms of the error
# between our prediction and the data:

##

# We are now ready to optimise. This will take some time so to avoid boredom
# and to understand how well the optimisation is working we will plot the
# model prediction of $g$ as we run the optimiser. To do this we provide
# a simple callback. This probably slows down the optimisation but is useful
# for us to see, and probably useful in practice to tell when the optimisation is
# stuck:

##

# We now setup the optimisation and run it 200 times:

##

# We didn't do very well. Let's try changing the optimiser, passing in the previous solution
# as the initial guess:

##

# This did much better and meets the ballpark norm.


# **Problem 1** Replace the neural network with a multilayer network and smooth activation
# function. Can you get better results than the simple RELU network?
## TODO: Construct a multilayer NN with smooth activation and see if it performs better



# **Problem 2** Use the predator-prey model
# $$
# \begin{bmatrix} x' \\ y' \end{bmatrix} =  \begin{bmatrix}αx - βxy \\  δxy - γy\end{bmatrix}
# $$
# on $T ∈ [0,5]$ with $α , β,δ,γ = 1,2,3,4$ with initial condition $[1,2]$
# to generate training data of samples at 21 evenly spaced points (only do a single run).
# Suppose we do not know the whole interaction but can model
# $$
#  \begin{bmatrix} x' \\ y' \end{bmatrix} =  \begin{bmatrix}αx \\ - γy\end{bmatrix} + g(x,y)
# $$
# where $g :ℝ^2 → ℝ^2$ is modeled by a Neural Network. Deduce $g$ by optimization of a loss when
# compared to the training data.
# Hint: This [SciML example](https://docs.sciml.ai/Overview/stable/showcase/missing_physics/)
# solves this problem and might help guide you.

## TODO: Learn the dynamics in a predator-prey model.



# ## 6.2 Neural ODEs

# A neural ODE is essentially the same as an ODE where the right-hand side
# is a neural network, i.e., we represent the solution to the ODE
# $$
# 𝐮' = f(𝐮)
# $$
# but where $f$ is given by a Neural Network.
# This is a more specialised version of
# the above but without the case where we incorporating parts of the model that are known.
# The idea is that they can be used as layers in more complicated Neural Networks.

# We can create a Neural ODE using the `NeuralODE` type in DiffEqFlux.jl
# (despite the name, DiffEqFlux.jl now uses Lux.jl instead of the older similar package Flux.jl):

##

# We can compare this to the same solve as an ODE problem:

##

# Unlike the solve as above we use Neural ODEs in Chains.
# Here's a simple artificial example:

##

# To understand why this is useful relates to how one designs neural networks to
# match the problem, something we're not going to dig into. But in the next section
# we show how it can be used to solve a fun real world problem.

# ## 6.3 Image classification via Neural ODEs

# In our final example, we're going to look at the go-to problem of classifying numbers
# given an image, specified as pixels, as in the MNIST database.
# We are going to walk through one of the [standard examples](https://docs.sciml.ai/DiffEqFlux/stable/examples/mnist_neural_ode/)
# in DiffEqFlux.jl but simplified (the original example supports GPUs).

# First let's load the database:

##

# This is a database which contains an image (in the "features" key)
# and what number that image represents (in the "targets" key).

##

# This is a database of 60k images.
# We can see an example here:

##

# We can plot the image by converting the elements to `Gray` in which case
# Images.jl automatically plots. We transpose the pixels since the default
# has $x$ and $y$ axes swapped:

##

# We can recognise this is a 6, and the database tells us this information:

##

# We want to create a NN with a Neural ODE layer to approximate the map
# from a 28 × 28 image (represented by `Float64`) to a number.
# But having the output be a number isn't quite enough since we can only be approximately
# accurate. Therefore we want to map from an image to a 10-vector where the
# entry with the largest value is the number (+1), and other entries somehow give us extra information
# about the chance that its that number. Eg. we want to use a
# so-called "one hot" encoding of the number. We can do so with the following function:


##

# Thus we want to construct a map from an image to a 10-vector and we represent
# this map by a NN, one with a NeuralODE layer. Without digging into the motivation
# we will follow the example.

# First we want to "downsample" an image from $ℝ^{28 × 28}$
# to $ℝ^{20}$, but we also want to work with $N$ images at the same time
# for efficiency reasons ("batching").
# The first step is to flatten matrices to vectors using a `FlattenLayer`
# which is map from $ℝ^{28 × 28 × N} → ℝ^{28^2 × N}$ by just vectorising the
# images by columns:

##

# We now combine it with a Neural network to get a map ${\rm down} : ℝ^{28 × 28 × N} → ℝ^{20 × N}$:

##

# The next layer is going to be a map from initial  conditions to final values
# $ℝ^{20 × N} → ℝ^{20 × N}$ where we feed
# each of the outputs into a Neural ODE whose RHS is given by another NN:

##

# Since our output needs to be a 10-vector we finally pass it through one last layer
# that down samples $ℝ^{20 × N} → ℝ^{10 × N}$:

##

# We can put everything together. A nice feature of Lux.jl is we can give names to our
# layers (more descriptive than `layer_1`) as follows (here the `convert` layer maps a solution
# to an ODE to its final value which we need to wrap in order for it to work with Lux):

##

# Our NN is pretty fast! But the parameters to actually give us the right out:

##

# So now we want to setup a loss function to choose the parameters. 
# This will be based on matching data.
# For efficiency we need to group our data into batches:

##


# We want to choose the parameters in our model to map `train[b][1]` to `train[b][2]`
# for every batch `b`.
# But we simply want to measure the largest components are in the same spot, not necessarily
# that the $k$-th entry is close to 1 and all other entries are close to 0.
# Statistics/information theory tells us that  following gives us a good loss function
# for imposing this (which is beyond my ken):

##


# We can see how our random parameters do:

##

# We can now optimise the paramters to fit the data. Let's setup the optimisation problem:

##

# We'll make a callback to measure the progress:

##

# Finally we can train the NN-ODE and monitor the loss and weights.

##

# Did it work? Let's try an image in the database:

##

# Certainly not a "onehot" vector! But we can recover the predicted value:

##

# It was correct! But what about the images not in our training set?

##

# This may or may not be correct depending on the training:

##

# But we can find the percentage it gets right:

##

# >90% accuracy is pretty good!

# ## 6.4 Final remarks

# We have only dipped our toes into the very basics of SciML.
# The [SciML website](https://sciml.ai) is a good place for more serious examples.

# There is also a closely related area of Physics Informed Neural Networks (PINNs)
# where one tries to replace ODE solvers with NNs by training them on input-output pairs.
# Based on what we have seen, I am highly sceptical this is useful for low dimensional problems
# where we have very good numerical methods that not only achieve "eyeball norm" but sometimes
# much more accuracy (as much as even 16 digits!).
# But for high-dimensional problems the number classification  problem gives some indication
# that these methods can play a role for producing "ballpark norm" approximations.
# Unfortunately its much harder to do a sanity check for problems where we can't necessarily
# look at the output and say "yeah, that's an 8 and the algorithm thinks its an 8".

# I think a bigger impact of this technology is the robustness of automatic differentiation
# which can play a big role in classical applied mathematics without necessarily using neural
# networks. I think the future of numerical algorithms will very much consider the ability to
# perform automatic differentiation efficiently. The SciML developers, in particular Chris Rackackas,
# have also realised that the role of stiff versus non-stiff and stability of forward versus reverse
# automatic differentiation are intrinsically linked, so there is a big need for classical numerical analysis
# even in this brave new world.