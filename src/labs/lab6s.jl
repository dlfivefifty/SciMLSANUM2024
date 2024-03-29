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
# First let's setup some training data with different initial conditions:


function firstorder_rhs!(du, u, α, t)
    du[1] = u[1] - α*u[1]^3
end

N_trials = 20
t = range(0, 10; length=50)
data = zeros(length(t), N_trials)
α = 2.3
for j = 1:N_trials
    prob = ODEProblem(firstorder_rhs!, [randn()], (0.0, 10), α)
    data[:,j] = Vector(solve(prob; saveat=t))
end

scatter(t, data)


# We will now try to deduce the term by training a simple neural network.





n = 100

const RELU_MODEL = Chain(Dense(1 => n, relu), Dense(n => 1))
rng =  MersenneTwister()
ps,st = Lux.setup(rng, RELU_MODEL)
const RELU_ST = st
ps = ComponentArray(ps);

function firstorder_rhs_nn!(du, u, p, t)
    du[1] = u[1]  + RELU_MODEL(u, p, RELU_ST)[1][1]
end

function firstorder_loss(p, (data, t))
    loss = 0.0
    for j = 1:size(data,2)
        prob = ODEProblem(firstorder_rhs_nn!, data[1:1,j], (0.0, 10), p)
        y = solve(prob, Vern7(), abstol = 1e-6, reltol = 1e-6, saveat=t)
        loss += norm(data[:,j] - Vector(y))
    end
    loss
end

prob = OptimizationProblem(OptimizationFunction(firstorder_loss, Optimization.AutoZygote()), ps, (data, t))

losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

@time ret = solve(prob, Adam(0.03), maxiters=300, callback=callback)
g = range(-1,1;length=100)
plot(g,-α*g.^3)
plot!(g,RELU_MODEL(g', ret.u, RELU_ST)[1]')

prob = OptimizationProblem(OptimizationFunction(firstorder_loss, Optimization.AutoZygote()), ret.u, (data, t))
@time ret = solve(prob, Adam(0.03), maxiters=300, callback=callback)


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



# **Problem** Consider the predator-prey model but where we don't know the 
# Hint: This [SciML example](https://docs.sciml.ai/Overview/stable/showcase/missing_physics/)
# solves this problem and might help guide you.

## SOLUTION
## This is modified from the above. 

## END

## 6.2 Neural differential equations in DiffEqFlux.jl



