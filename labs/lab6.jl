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

using Lux, DifferentialEquations, Optimization, OptimizationOptimisers, Plots, Zygote, SciMLSensitivity, ComponentArrays, Random, LinearAlgebra, Test


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



