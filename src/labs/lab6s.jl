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


function firstorder_rhs!(du, u, α, t)
    du[1] = u[1] - α*u[1]^3
end

## Provide a random number generator for reliability (and so data covers large range of possible $u$ values)
rng =  MersenneTwister(2121)
α = 2.3 # arbitrary scaling
N_trials = 15
t = range(0, 5; length=15)
data = zeros(length(t), N_trials)
for j = 1:N_trials
    u₀ = randn(rng) # random initial condition
    prob = ODEProblem(firstorder_rhs!, [u₀], (0.0, t[end]), α)
    data[:,j] = Vector(solve(prob; saveat=t))
end

scatter(t, data; legend=false) # plot the data


# We will now try to deduce the term $-αu^3$ by training a simple NN
# by minimising the error when comparing the model to the provided data.
# Because Optimzation.jl (currently) requires that parameters behave like
# arrays, rather than passing in the NN as a parameter we make it
# a global constant. We begin with simple 2-layer piecewise affine NN:


n = 100
const RELU_MODEL = Chain(Dense(1 => n, relu), Dense(n => 1))

ps,st = Lux.setup(rng, RELU_MODEL)
const RELU_ST = st # RELU_ST is "no state", make it a constant
ps = ComponentArray(ps); # Convert our parameters to an AbstractArray

# Our model is
# $$
#   u' = u + g(u)
# $$
# where we represent $g$ by a NN with given parameters. Here is the rhs for this simple model:

function firstorder_rhs_nn!(du, u, p, t)
    du[1] = u[1]  + RELU_MODEL(u, p, RELU_ST)[1][1]
end

# We can then compute the loss by solving the ODE with a given set of parameters
# for each of the runs in our samples and summing over the 2-norms of the error
# between our prediction and the data:

function firstorder_loss(p, (data, t))
    loss = 0.0
    for j = 1:size(data,2)
        prob = ODEProblem(firstorder_rhs_nn!, data[1:1,j], (0.0, t[end]), p)
        pred = solve(prob, Vern7(), abstol = 1e-6, reltol = 1e-6, saveat=t)
        loss += norm(data[:,j] - Vector(pred))
    end
    loss
end

# We are now ready to optimise. This will take some time so to avoid boredom
# and to understand how well the optimisation is working we will plot the
# model prediction of $g$ as we run the optimiser. To do this we provide
# a simple callback. This probably slows down the optimisation but is useful
# for us to see, and probably useful in practice to tell when the optimisation is
# stuck:


relu_callback = function (p, l)
    g = range(-1,1;length=30)
    pred =  RELU_MODEL(g', p.u, RELU_ST)[1]'
    plt = plot(g, -2.3*g.^3; label="true")
    plot!(plt, g, pred; label = "prediction", title="loss: $l")
    display(plt)
    return false
end

# We now setup the optimisation and run it 200 times:

prob = OptimizationProblem(OptimizationFunction(firstorder_loss, AutoZygote()), ps, (data, t))
@time ret = solve(prob, Adam(0.03), maxiters=200, callback=relu_callback)

# We didn't do very well. Let's try changing the optimiser, passing in the previous solution
# as the initial guess:

using OptimizationOptimJL # Load LBFGS optimiser
prob = OptimizationProblem(OptimizationFunction(firstorder_loss, AutoZygote()), ret.u, (data, t))
@time ret = solve(prob, LBFGS(), maxiters=200, callback=relu_callback)

# This did much better and meets the ballpark norm.


# **Problem 1** Replace the neural network with a multilayer network and smooth activation
# function. Can you get better results than the simple RELU network?
## TODO: Construct a multilayer NN with smooth activation and see if it performs better

## SOLUTION

smoothstep = x -> x*(tanh(10x) + 1)/2

## Multilayer model
const SMOOTHSTEP_MODEL = Chain(Dense(1, 5, smoothstep), Dense(5, 5, smoothstep), Dense(5, 5, smoothstep),
              Dense(5, 1))
## Get the initial parameters and state variables of the model
ps, st = Lux.setup(rng, SMOOTHSTEP_MODEL); ps = ComponentArray(ps)
const SMOOTHSTEP_ST = st

function firstorder_rhs_smoothstep!(du, u, p, t)
    du[1] = u[1]  + SMOOTHSTEP_MODEL(u, p, SMOOTHSTEP_ST)[1][1]
end

function firstorder_loss_smoothstep(p, (data, t))
    loss = 0.0
    for j = 1:size(data,2)
        prob = ODEProblem(firstorder_rhs_smoothstep!, data[1:1,j], (0.0, t[end]), p)
        pred = Vector(solve(prob, Vern7(), abstol = 1e-6, reltol = 1e-6, saveat=t))
        loss += norm(data[:,j] - pred)
    end
    loss
end



smoothstep_callback = function (p, l)
    g = range(-1,1;length=30)
    pred =  SMOOTHSTEP_MODEL(g', p.u, SMOOTHSTEP_ST)[1]'
    plt = plot(g, -2.3*g.^3; label="true")
    plot!(plt, g, pred; label = "prediction", title="loss: $l")
    display(plt)
    return false
end

prob = OptimizationProblem(OptimizationFunction(firstorder_loss_smoothstep, AutoZygote()), ps, (data, t))
@time ret = solve(prob, Adam(0.03), maxiters=300, callback=smoothstep_callback)

prob = OptimizationProblem(OptimizationFunction(firstorder_loss_smoothstep, AutoZygote()), ret.u, (data, t))
@time ret = solve(prob, LBFGS(), maxiters=200, callback=smoothstep_callback)

prob = OptimizationProblem(OptimizationFunction(firstorder_loss_smoothstep, AutoZygote()), ret.u, (data, t))
@time ret = solve(prob, LBFGS(), maxiters=200, callback=smoothstep_callback)

## I can't get better results!  😅

## END

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

## SOLUTION
## This is modified from the above link.

function lotka!(du, u, p, t)
    x,y = u
    α, β, γ, δ = p
    du[1] = α * x - β * y * x
    du[2] = γ * x * y - δ * y
end

## Define the experimental parameter
u₀ = [1,2]
p_ = [1,2,3,4]
prob = ODEProblem(lotka!, u0, (0.0, 5.0), p_)
t = range(0, 5; length=21)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = t)
plot(solution)
#
X = Array(solution)

rbf(x) = exp.(-(x .^ 2))

## Multilayer FeedForward
const RBF_MODEL = Chain(Dense(2, 5, rbf), Dense(5, 5, rbf), Dense(5, 5, rbf), Dense(5, 2))
## Get the initial parameters and state variables of the model
ps, st = Lux.setup(rng, RBF_MODEL); ps = ComponentArray(ps)
const RBF_ST = st

## Define the hybrid model
function ude_dynamics_rhs!(du, u, p, t)
    û = RBF_MODEL(u, p, RBF_ST)[1] # Network prediction
    du[1] = u[1] + û[1]
    du[2] = -2u[2] + û[2]
end

## Define the problem

function ude_solve(p)
    prob = ODEProblem(ude_dynamics_rhs!, [1.,2.], (0.0, 5.0), p)
    solve(prob, Vern7(), abstol = 1e-6, reltol = 1e-6, saveat=t)
end

function ude_loss(p, (data, t))
    pred = Array(ude_solve(p))
    norm(data - pred)
end


ude_callback = function (p, l)
    display(plot(ude_solve(p.u); title="loss = $l"))
    return false
end

prob = OptimizationProblem(OptimizationFunction(ude_loss, AutoZygote()), ps, (X, t))
@time ret = solve(prob, Adam(0.03), maxiters=5000, callback=ude_callback)

prob = OptimizationProblem(OptimizationFunction(ude_loss, AutoZygote()), ret.u, (X, t))
@time ret = solve(prob, LBFGS(), maxiters=2000, callback=ude_callback)


## END

## 6.2 Neural ODEs

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

using DiffEqFlux
n_ode = NeuralODE(RELU_MODEL, (0., 5.0))
p,st = Lux.setup(rng, n_ode)
p = ComponentArray(p)

u₀ = 0.1
u = n_ode([u₀], p, st)[1] # returns a solution to an ODE
plot(u)

# We can compare this to the same solve as an ODE problem:

function n_ode_rhs!(du, u, p, t)
    du[1] = RELU_MODEL(u, p, RELU_ST)[1][1]
end

prob = ODEProblem(n_ode_rhs!, [u₀], (0.0, 5.0), p)
ũ = solve(prob)
plot!(ũ)

# Unlike the solve as above we use Neural ODEs in Chains.
# Here's a simple artificial example:

nn_chain_model = Chain(Dense(5 => 1, relu), n_ode)
p, st = Lux.setup(rng, nn_chain_model)
plot(nn_chain_model(randn(5), p, st)[1]) # Initial condition given by a Dense layer

# To understand why this is useful relates to how one designs neural networks to
# match the problem, something we're not going to dig into. But in the next section
# we show how it can be used to solve a fun real world problem.

## 6.3 Image classification via Neural ODEs

# In our final example, we're going to look at the go-to problem of classifying numbers
# given an image, specified as pixels, as in the MNIST database.
# We are going to walk through one of the [standard examples](https://docs.sciml.ai/DiffEqFlux/stable/examples/mnist_neural_ode/)
# in DiffEqFlux.jl but simplified (the original example supports GPUs).

#First let's load the database:

using Images # Useful for plotting images given by pixels.
using MLDatasets: MNIST # load the MNIST function
mnist = MNIST() # loads the database

# This is a database which contains an image (in the "features" key)
# and what number that image represents (in the "targets" key).

imgs, nums = mnist.features, mnist.targets
size(imgs), length(nums)

# This is a database of 60k images.
# We can see an example here:

l = 21_235 # which of the 60k images
imgs[:,:,l] # numbers between 0 and 1 representing greyscale

# We can plot the image by converting the elements to `Grey` in which case
# Images.jl automatically plots. We transpose the pixels since the default
# has x and y swapped:

Gray.(imgs[:,:,l])'

# We can recognise this is a 6, and the database tells us this information:

nums[l]

# We want to create a NN with a Neural ODE layer to approximate the map
# from a 28 × 28 image (represented by `Float64`) to a number.
# But having the output be a number isn't quite enough since we can only be approximately
# accurate. Therefore we want to map from an image to a 10-vector where the
# entry with the largest value is the number (+1). Eg. we want to use a
# so-called "one hot" encoding of the number. We can do so with the following function:



function onehot(nums::AbstractVector)
    n = length(nums)
    ret = zeros(Int,10,n)
    for j = 1:n
        ret[nums[j]+1,j] = 1
    end
    ret
end

onehot(nums[l:l]) # (6+1)th entry is 1 all else are zero

# Thus we want to construct a map from an image to a 10-vector and we represent
# this map by a NN, one with a NeuralODE layer. Without digging into the motivation
# we will follow the example.

# First we want to "downsample" an image from $ℝ^{28 × 28}$
# to $ℝ^{20}$, but we also want to work with $N$ images at the same time
# for efficiency reasons ("batching").
# The first step is to flatten matrices to vectors using a `FlattenLayer`:

@test FlattenLayer()(imgs[:,:,1:3], NamedTuple(), NamedTuple())[1] == [vec(imgs[:,:,1]) vec(imgs[:,:,2]) vec(imgs[:,:,3])]

# We now combine it with a Neural network to get a map ${\rm down} : ℝ^{28 × 28 × N} → ℝ^{20 × N}$
# this composed of a `FlattenLayer` (that maps  matrices $ℝ^{28 × 28}$ to vectors $ℝ^{28^2}$
# by vectorising):


down = Chain(FlattenLayer(), Dense(784, 20, tanh))
down_p, down_st = Lux.setup(rng, down)
down(imgs[:,:,1:3], down_p, down_st)[1] # Each column is a different image

# The next layer is going to be a map from initial  conditions to final values
# $ℝ^{20 × N} → ℝ^{20 × N}$ where we feed
# each of the outputs into a Neural ODE whose RHS is given by another NN:

nn = Chain(Dense(20, 10, tanh), Dense(10, 10, tanh), Dense(10, 20, tanh)) # RHS to ODE

nn_ode = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false, reltol = 1e-3,
    abstol = 1e-3, save_start = false)

# Since our output needs to be a 10-vector we finally pass it through one last layer
# that down samples $ℝ^{20 × N} → ℝ^{10 × N}$:

fc = Dense(20, 10)

# We can put everything together. A nice feature of Lux.jl is we can give names to our
# layers (more descriptive than `layer_1`) as follows (here the `convert` layer maps a solution
# to an ODE to its final value which we need to wrap in order for it to work with Lux):
m = Chain(; down=down, nn_ode=nn_ode, convert = WrappedFunction(last), fc)
ps, st = Lux.setup(rng, m)
ps = ComponentArray(ps)

# Our NN is pretty fast! But the parameters to actually give us the right out:

@time m(imgs[:,:,1:5], ps, st)[1] # We want this to == onehot(nums[1:5])

# So now we want to setup a loss function to choose the parameters. 
# This will be based on matching data.
# For efficiency we need to group our data into batches:

const N = 160 # batch together 160 images at a time

M = length(nums) ÷ N - 1# number of batches.
## We don't do the last batch so we can see how well it performs.
data = [(imgs[:,:,N*(b-1)+1:N*b],onehot(nums[N*(b-1)+1:N*b])) for b=1:M]


# We want to choose the parameters in our model to map `train[b][1]` to `train[b][2]`
# for every batch `b`.
# But we simply want to measure the largest components are in the same spot, not necessarily
# that the $k$-th entry is close to 1 and all other entries are close to 0.
# Statistics/information theory tells us that  following gives us a good loss function
# for imposing this (which is beyond my ken):

logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims = 1); dims = 1))

function loss_function(ps, x, y)
    pred, st_ = m(x, ps, st)
    return logitcrossentropy(pred, y), pred
end


# We can see how our random parameters do:

@time loss_function(ps, data[1]...)

# We can now optimise the paramters to fit the data. Let's setup the optimisation problem:

opt_func = OptimizationFunction((ps, _, x, y) -> loss_function(ps, x, y), AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps)

# We'll make a callback to measure the progress:

iter = 0
function callback(ps, l, pred)
    global iter += 1
    if (iter % 10 == 0)
        @info "loss = $l"
    end
    return false
end

# Finally we can train the NN-ODE and monitor the loss and weights.
res = solve(opt_prob, Adam(0.05), data; callback)

# Did it work? Let's try an image in the database:

l = 22352
Gray.(imgs[:,:,l])'
m(imgs[:,:,l:l], res.u, st)[1]

# Certainly not a "onehot" vector! But we can recover the predicted value:

classify(x) = [argmax(x[:,j]) - 1 for j = 1:size(x,2)]
classify(m(imgs[:,:,l:l], res.u, st)[1])

# It was correct! But what about the images not in our training set?

l = 60_000
Gray.(imgs[:,:,l])'

# And again it correct!
classify(m(imgs[:,:,l:l], res.u, st)[1])

# In fact we can find the percentage it gets right:

sum(classify(m(imgs[:,:,end-N+1:end], res.u, st)[1]) .== nums[end-N+1:end])/N

# 93% accuracy is pretty good!

# 6.4 Future directions

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