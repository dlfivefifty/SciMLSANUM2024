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

## 6.2 Neural ODEs and number classification


using Images
using MLDatasets: MNIST

mnist = MNIST()
imgs, labels_raw = mnist.features, mnist.targets
l = 21235
Gray.(imgs[:,:,l])
labels_raw[l]

using DiffEqFlux, MLDataUtils, NNlib
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs


logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims = 1); dims = 1))

function loadmnist(batchsize = bs)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    function onehot(labels_raw)
        convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
    end
    # Load MNIST
    mnist = MNIST()
    imgs, labels_raw = mnist.features, mnist.targets
    # Process images into (H,W,C,BS) batches
    x_train = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    x_train = batchview(x_train, batchsize)
    # Onehot and batch the labels
    y_train = onehot(labels_raw)
    y_train = batchview(y_train, batchsize)
    return x_train, y_train
end



# Main
const bs = 128
x_train, y_train = loadmnist(bs)

down = Chain(FlattenLayer(), Dense(784, 20, tanh))
nn = Chain(Dense(20, 10, tanh), Dense(10, 10, tanh),
    Dense(10, 20, tanh))
fc = Dense(20, 10)

nn_ode = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false, reltol = 1e-3,
    abstol = 1e-3, save_start = false)

DiffEqArray_to_Array(x) = reshape(x, size(x)[1:2])

#Build our over-all model topology
m = Chain(; down=down, nn_ode=nn_ode, convert = Lux.WrappedFunction(DiffEqArray_to_Array), fc)
ps, st = Lux.setup(Random.default_rng(), m)
ps = ComponentArray(ps)

#We can also build the model topology without a NN-ODE
m_no_ode = Chain(; down, nn, fc)
ps_no_ode, st_no_ode = Lux.setup(Random.default_rng(), m_no_ode)
ps_no_ode = ComponentArray(ps_no_ode)

#To understand the intermediate NN-ODE layer, we can examine it's dimensionality
x_d = first(down(x_train[1], ps.down, st.down))

# We can see that we can compute the forward pass through the NN topology featuring an NNODE layer.
x_m = first(m(x_train[1], ps, st))
#Or without the NN-ODE layer.
x_m = first(m_no_ode(x_train[1], ps_no_ode, st_no_ode))

classify(x) = argmax.(eachcol(x))

function accuracy(model, data, ps, st; n_batches = 100)
    total_correct = 0
    total = 0
    st = Lux.testmode(st)
    for (x, y) in collect(data)[1:n_batches]
        target_class = classify(y)
        predicted_class = classify(first(model(x, ps, st)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end
#burn in accuracy
accuracy(m, zip(x_train, y_train), ps, st)

function loss_function(ps, x, y)
    pred, st_ = m(x, ps, st)
    return logitcrossentropy(pred, y), pred
end

#burn in loss
loss_function(ps, x_train[1], y_train[1])

opt = OptimizationOptimisers.Adam(0.05)
iter = 0

opt_func = OptimizationFunction((ps, _, x, y) -> loss_function(ps, x, y),
    Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps)

function callback(ps, l, pred)
    global iter += 1
    #Monitor that the weights do infact update
    #Every 10 training iterations show accuracy
    if (iter % 10 == 0)
        @info "[MNIST GPU] Accuracy: $(accuracy(m, zip(x_train, y_train), ps, st))"
    end
    return false
end

# Train the NN-ODE and monitor the loss and weights.
res = Optimization.solve(opt_prob, opt, zip(x_train, y_train); callback)
@test accuracy(m, zip(x_train, y_train), res.u, st) > 0.8

m(x_train[1][:,:,:,:], res.u, st)[1]
Gray.(x_train[1][:,:,1,1])

Gray.(imgs[:,:,l])



classify(m(reshape(imgs[:,:,l], 28, 28, 1), res.u, st)[1])[1]-1