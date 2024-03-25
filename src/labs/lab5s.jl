# # SciML SANUM2024
# # Lab 5: Neural Networks and Lux.jl
#
# In this lab we introduce neural networks as implemented in Lux.jl. 
# A neural network (NN) is in some sense just a function with many parameters,
# in a way that facilitates computing gradients with respect to these parameters.
# It is constructed by composing basic building blocks usually built from linear
# algebra operations, combined with simple _activator functions_. 
# Here we look at the simplest case and see how the paremeters in a NN can be chosen to
# solve optimisation problems. 

# **Learning Outcomes**
# 1. Single-layer neural networks and activation functions.
# 2. Creating deeper networks as a `Chain`.
# 3. Training neural networks by simple optimisation.

using Lux, Random, Optimisers, Zygote, Plots, Test


# ## Single layer neural networs

# We begin with a single-layer neural network without an activator
# function. This happens to be precisely maps of the form
# $$
# 𝐱 ↦ A𝐱 + 𝐛
# $$
# where $A ∈ ℝ^{m × n}$ and $𝐛 ∈ ℝ^n$. The space of such maps is
# modelled by the `Dense` type, where the `weight` corresponds to $A$
# and the `bias` corresponds to $𝐛$. Here we see a simple example
# of constructing the model (the space of all such maps) and evaluating
# a specific map by specifying the paramters:

m,n = 5,4

model = Dense(n => m) # represents

A = randn(5,4)
b = randn(5)
x = randn(4)
st = NamedTuple() # no state
val,newst = model(x, (weight=A, bias=b), st) # returns the output of the map and the "state", which we ignore

@test val ≈ A*x + b # our model with these parameters is just A*x + b


# An important feature is that we can compute gradients with respect to parameters of functions of our
# model. Before we looked at the case where
# we differentiated with respect to vectors but a power feature in Zygote is it works for all types like named-tuples.


ps = (weight=A, bias=b)
ps_grad = gradient(p -> sum(model(x, p, st)[1]), ps)[1] # returns a named tuple containing the gradients

# Because our Neural Network at this stage is linear in the paremeters the gradient is actually quite simple: eg the partial derivative with
# respect to $A[k,j]$ will just be $x[j]$ and the derivative with respect to $b[k]$ will just be $1$. Thus we get:


@test ps_grad[:weight] ≈ ones(5) * x'
@test ps_grad[:bias] ≈ ones(5)




# Going beyond basic linear algebra, we can apply an "activator" function $f$ to each
# entry of the map, to represent maps of the form:
# $$
# 𝐱 ↦ f.(A𝐱 + 𝐛)
# $$
# Where we use the Julia-like broadcast notation to mean entrywise application.
# The classic in ML is the `relu` function which is really just $\max(0,x)$:

x = range(-1,1, 1000)
plot(x, relu.(x); label="relu")

# We can incorporate this in our model as follows:

model = Dense(4 => 5, relu)
model(x, (weight = A, bias=b), st)[1]

# And we can differentiate:

ps = (weight=A, bias=b)
ps_grad = gradient(p -> sum(model(x, p, st)[1]), ps)[1] # returns a named tuple containing the gradients

# **Problem** Derive the forumula  for the gradient of the model with an activator function and compare it with
# the numerical result just computed. Hint: The answer depends on the output value.

## SOLUTION
## the partial derivative with
## respect to $A[k,j]$ will just be $x[j]$ and the derivative with respect to $b[k]$ will just be $1$. Thus we get:
## END

# Let's see an example directly related to a classic numerical analysis problem: approximating functions by a continuous piecewise affine
# function, as done in the Trapezium rule. Unlike a standard approximation we do not 



n = 2
model = Dense(1 => n, relu)
A = randn(n,1)
b = randn(n)
st = NamedTuple() # no state
p = x -> sum(model([x], (weight = [1;2;;], bias=[3,-1]), st)[1])
x = range(-5,5, 100_00)
plot(x, p.(x))

n = 10
model = Chain(Dense(n => 1), Dense(1 => n, relu))
A = randn(n,1)
b = randn(n)
st = NamedTuple() # no state
p = x -> sum(model([x], (weight = A, bias=b), st)[1])
x = range(-5,5, 100_00)
plot(x, p.(x))




model = Chain(Dense(1 => n, relu), Dense(n => 1))
A = randn(n,1)
B = randn(1,n)
c = [0.0]
st = (layer_1 = NamedTuple(), layer_2 = NamedTuple())
p = x -> sum(model([x], (layer_1 = (weight=A, bias=b),
                         layer_2 = (weight=B, bias=c)), st)[1])

x = range(-5,5, 100_00)
plot(x, p.(x))

rng = MersenneTwister()

Random.seed!(rng, 12345)
opt = Adam(0.03f0)
tstate = Lux.Training.TrainState(rng, model, opt)
model([0.1], tstate.parameters, tstate.states)