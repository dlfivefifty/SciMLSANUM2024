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

using Lux, DifferentialEquations, Optimization, OptimizationOptimisers, Plots, Zygote, SciMLSensitivity, ComponentArrays, Random, Test


# 6.1 Learning dynamics

# We begin with a very simple ODE:
# $$
# u' = u - α sin(u)
# $$
# where we know $f(u) = u$ but suppose we don't know $g(u) = -α u^2$. 
# First let's setup some training data with different initial conditions.
# We will do 10 trials which are sampled at 15 points for $t ∈ [0,5]$.


function firstorder_rhs!(du, u, α, t)
    du[1] = u[1] - α*u[1]^3
end

N_trials = 10
t = range(0, 5; length=15)
data = zeros(length(t), N_trials)
α = 2.3
for j = 1:N_trials
    prob = ODEProblem(firstorder_rhs!, [randn()], (0.0, t[end]), α)
    data[:,j] = Vector(solve(prob; saveat=t))
end

scatter(t, data)


# We will now try to deduce the term $-αu^3$ by training a simple NN
# by minimising the error when modeling the provided data. 
# Because Optimzation.jl (currently) requires that parameters behave like
# arrays, rather than passing in the neural network as a parameter we make it
# a global constant. We begin with simple 2-layer piecewise affine NN:


n = 100
const RELU_MODEL = Chain(Dense(1 => n, relu), Dense(n => 1))
rng =  MersenneTwister() # need to provide a random number generator
ps,st = Lux.setup(rng, RELU_MODEL)
const RELU_ST = st # RELU_ST is "no state", make it a constant
ps = ComponentArray(ps); # Convert our parameters to an AbstractArray

# Our model is
# $$
#   u' = u + g(u)
# $$
# where we represent $g$ by a NN. Here is the rhs for this simple model:

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

# We are now ready to optimise. This will take some time so to avoid boredum 
# and to understand how well the optimisation is working we will plot dyna


callback = function (p, l)
    g = range(-1,1;length=30)
    plt = plot(g, -2.3*g.^3; label="true")
    plot!(plt, g, RELU_MODEL(g', p.u, RELU_ST)[1]'; label = "prediction", title="loss: $l")
    display(plt)
    return false
end

prob = OptimizationProblem(OptimizationFunction(firstorder_loss, Optimization.AutoZygote()), ps, (data, t))
@time ret = solve(prob, Adam(0.03), maxiters=200, callback=callback)

# We didn't do very well. Let's try changing the optimiser, passing in the previous solution:

prob = OptimizationProblem(OptimizationFunction(firstorder_loss, Optimization.AutoZygote()), ret.u, (data, t))
@time ret = solve(prob, LBFGS(), maxiters=200, callback=callback)



# ** Problem 1** Replace the neural network with a multilayer network and smooth activation
# function and train until the loss is better than the above example.


rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
const RBF_MODEL = Lux.Chain(Lux.Dense(1, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
              Lux.Dense(5, 1))
# Get the initial parameters and state variables of the model
ps, st = Lux.setup(rng, RBF_MODEL); ps = ComponentArray(ps)
const RBF_ST = st

function firstorder_rhs_nn!(du, u, p, t)
    du[1] = u[1]  + RBF_MODEL(u, p, RBF_ST)[1][1]
end


prob = OptimizationProblem(OptimizationFunction(firstorder_loss, Optimization.AutoZygote()), ps, (data, t))
@time ret = solve(prob, Adam(0.03), maxiters=2000, callback=callback)

ps = ret.u
plot(g,-α*g.^3)
plot!(g,RBF_MODEL(g', ps, RBF_ST)[1]')

prob = OptimizationProblem(OptimizationFunction(firstorder_loss, Optimization.AutoZygote()), ps, (data, t))
import Optim
using OptimizationOptimJL
@time ret = solve(prob, Optim.LBFGS(), maxiters=200, callback=callback)

ps = ret.u
plot(g,-α*g.^3)
plot!(g,RBF_MODEL(g', ps, RBF_ST)[1]')



# **Problem 2** Use the predator-prey model
# $$
# \begin{bmatrix} x' \\ y' \end{bmatrix} =  \begin{bmatrix}αx - βxy \\  δxy - γy\end{bmatrix}
# $$
# on $T ∈ [0,10]$ with $α , β,δ,γ = 1,2,3,4$ with initial condition $[1,2]$
# to generate training data (only a single run). 
# Suppose we do not know the whole interaction but can model
# $$
#  \begin{bmatrix} x' \\ y' \end{bmatrix} =  \begin{bmatrix}αx \\ - γy\end{bmatrix} + g(x,y)
# $$
# where $g :ℝ^2 → ℝ^2$ is modeled by a Neural Network. Deduce $g$ by optimization of a loss when
# compared to the training data.
# Hint: This [SciML example](https://docs.sciml.ai/Overview/stable/showcase/missing_physics/)
# solves this problem and might help guide you.

## SOLUTION
## This is modified from the above. 

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0, 5.0)
u0 = 5.0f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

plot(solution)
X = Array(solution)
t = solution.t

rbf(x) = exp.(-(x .^ 2))

## Multilayer FeedForward
const RBF_MODEL_2 = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
              Lux.Dense(5, 2))
## Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
const RBF_ST_2 = st

# Define the hybrid model
function ude_dynamics!(du, u, (p, (α,δ)), t)
    û = RBF_MODEL_2(u, p, RBF_ST_2)[1] # Network prediction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!, X[:, 1], tspan, p)

function ude_loss(p, (data, t))
    prob = ODEProblem(ude_dynamics!, data[1:1,j], (0.0, 10), p)
        y = solve(prob, Vern7(), abstol = 1e-6, reltol = 1e-6, saveat=t)
        loss += norm(data[:,j] - Vector(y))
    end
    loss
end

## END

## 6.2 Neural differential equations in DiffEqFlux.jl

# https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/


u0 = [2.0; 0.0]
t = range(0, 1.5; length=25)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, (0,1.5))
ode_data = Array(solve(prob_trueode, Tsit5(); saveat = t))

dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
p, st = Lux.setup(rng, dudt2)

prob_neuralode = NeuralODE(dudt2, (0,1.5), Tsit5(); saveat = t)

predict_neuralode(p) = Array(prob_neuralode(u0, p, st)[1])

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred)
    println(l)
    plt = scatter(t, ode_data[1, :]; label = "data")
    scatter!(plt, t, pred[1, :]; label = "prediction")
    display(plot(plt))
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob, Adam(0.05); callback = callback,
    maxiters = 300)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)

