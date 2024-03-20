# # SciML SANUM2024
# # Lab 5: Neural Networks and Lux.jl
#
# In this lab we introduce neural networks as implemented in Lux.jl. 
# A neural network (NN) is in some sense just a function with many parameters,
# in a way that facilitates computing gradients with respect to these parameters.
# It is constructed by composing basic building blocks usually built from linear
# algebra operations, combined with simple _activator functions_. 
# Here we look at the simplest case and see how the paremeters in a NN ca

using Lux, Random, Optimisers, Test



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


using Plots


n = 10
model = Dense(1 => n, relu)
A = randn(n,1)
b = randn(n)
p = x -> sum(model([x], (weight = A, bias=b), st)[1])
x = range(-5,5, 100_00)
plot(x, p.(x))



model = Chain(Dense(1 => n, relu), Dense(n => n, relu))
B = randn(n,n)
c = randn(n)
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