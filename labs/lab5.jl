# # SciML SANUM2024
# # Lab 5: Neural Networks and Lux.jl
#
# In this lab we introduce neural networks as implemented in Lux.jl. 
# A neural network (NN) is in some sense just a function with many parameters
# in a way that facilitates computing gradients with respect to these parameters.
# That is: it is at its core a way of book-keeping a heavily parameterised function.
# It is constructed by composing basic building blocks usually built from linear
# algebra operations, combined with simple _activator functions_. 
# Here we look at the simplest case and see how the paremeters in a NN can be chosen to
# solve optimisation problems. In other words: we will _train_ the NN.

# **Learning Outcomes**
# 1. Single-layer neural networks and activation functions.
# 2. Creating deeper networks as a `Chain`.
# 3. Training neural networks by simple optimisation.

using Lux, Random, Optimization, OptimizationOptimisers, ComponentArrays, Zygote, Plots, LinearAlgebra, Test


# ## 4.1 Single layer neural networks

# We begin with a single-layer NN without an activator
# function which correspond to maps of the form:
# $$
# 𝐱 ↦ A𝐱 + 𝐛
# $$
# where $A ∈ ℝ^{m × n}$ and $𝐛 ∈ ℝ^n$. The space of such maps is
# modelled by the `Dense` type which has two paramters: `weight`, corresponding to $A$, and
#  `bias`, corresponding to $𝐛$. Here we see a simple example
# of constructing the model (the space of all such maps) and evaluating
# a specific map by specifying the parameters.

# Note we have to pass an extra argument corresponding to
# the "state" of a NN: this doesn't exist for the simple layers we consider
# but more sophisticated NNs can depend on history and so the state records the relevant
# information.



m,n = 5,4

model = Dense(n => m) # represents maps of the form 𝐱 ↦ A𝐱 + 𝐛

A = randn(5,4)
b = randn(5)
x = randn(4)
const NOSTATE = NamedTuple() # no state for our NN
val,newst = model(x, (weight=A, bias=b), NOSTATE) # returns the output of the map and the updated state, which we ignore

@test val == A*x + b # our model with these parameters is just A*x + b



# An important feature is that we can compute gradients with respect to parameters of functions of our
# model. Before we looked at the case where
# we differentiated with respect to vectors but a powerful feature in Zygote is it works for other types, including the `NamedTuple`
# which Lux.jl uses for representing paramaters:


ps = (weight=A, bias=b) # parameters as a NamedTuple, almost like an anonymous type
ps_grad = gradient(p -> sum(model(x, p, NOSTATE)[1]), ps)[1] # returns a named tuple containing the gradients


# Because our NN at this stage is linear in the paremeters the gradient is actually quite simple: eg the partial derivative with
# respect to $A[k,j]$ will just be $x[j]$ and the derivative with respect to $b[k]$ will just be $1$. Thus we get:


@test ps_grad.weight ≈ ones(5) * x'
@test ps_grad.bias ≈ ones(5)




# Going beyond basic linear algebra, we can apply an "activator" function $f$ to each
# entry of the map, to represent maps of the form:
# $$
# 𝐱 ↦ f.(A𝐱 + 𝐛)
# $$
# where we use the Julia-like broadcast notation to mean entrywise application.
# The classic activator is the `relu` function which is really just $\max(0,x)$:


x = range(-1,1, 1000)
plot(x, relu.(x); label="relu")


# We can incorporate this in our model as follows:


x = randn(4)
model = Dense(4 => 5, relu)
model(x, (weight = A, bias=b), NOSTATE)[1]


# And we can compute gradients as before:


ps = (weight=A, bias=b)
ps_grad = gradient(p -> sum(model(x, p, NOSTATE)[1]), ps)[1] # returns a named tuple containing the gradients


# **Problem 1** Derive the formula  for the gradient of the model with an activator function and compare it with
# the numerical result just computed. Hint: The answer depends on the output value.

## TODO: Compute the gradient by hand, matching ps_grad


# Let's see an example directly related to a classic numerical analysis problem: approximating 
# functions by a continuous piecewise affine
# function, as done in the Trapezium rule. Our model corresponds to a sum of weighted and shifted `relu` functions:
# $$
# p_{𝐚,𝐛}(x) := ∑_{k=1}^n {\rm relu}(a_k x + b_k)
# $$
# We note that this is a sum of positive convex functions so will only be useful for approximating positive convex functions
# (we will generalise this later).  Thus we want to choose the paremeters to fit data generated by a positive convex function,
# e.g., $f(x) = \exp(x)$. Here we first generate "training data" which means the samples of the function on a grid.


n = 100
x = range(-1, 1; length = n)
y = exp.(x)
plot(x, y)


# Our one-layer NN (before the summation) is
# $$
#   {\rm relu}.(𝐚x + 𝐛)
# $$
# which corresponds to a simple dense layer with `relu` activation.
# We then sum over the output of this to get the model
# $$
#   [1,…,1]^⊤ {\rm relu}.(𝐚x + 𝐛)
# $$
# In our case `x` is actually a vector containing the grid we sample on
# but we first need to transpose it to be a $1 × n$ matrix, which will apply the NN
# to each grid point. We can then sum over the columns to get the value of the model with the given
# parameters at the grid points.


nn = Dense(1 => n, relu)
function summation_model(ps, (nn, x))
    Y,st = nn(x', ps, NOSTATE) # k-th column contains relu.(𝐚x[k] + 𝐛)
    vec(sum(Y; dims=1)) # sums over the columns
end


# We want to choose the parameters to minimise a loss function. Here we
# just wish to minimise the 2-norm error which we can write as follow:


function convex_regression_loss(ps, (nn, (x,y)))
    ỹ = summation_model(ps, (nn, x))
    norm(ỹ - y) # 2-norm error
end



# We now setup the optimation problem. We can use `Lux.setup` to create a random initial guess for parameters
# though we need to supply a random number generator. We also need to wrap the returned named tuple in a `ComponentArray` as Optimization.jl
# requires the optimisation to be over an array-type.


rng = MersenneTwister() # Random number generator.
ps = ComponentArray(Lux.setup(rng, nn)[1]) 

prob = OptimizationProblem(OptimizationFunction(convex_regression_loss, Optimization.AutoZygote()), ps, (nn, (x, y)))
@time ret = solve(prob, Adam(0.03), maxiters=250)

plot(x, y)
plot!(x, summation_model(ret.u, (nn, x)))



# **Problem 2**  Replace `relu` in the activation function with a smooth `tanh` function and plot
# the result. Is the approximation as accurate? What if you increase the number of epochs?
# What if you construct your own function that is a smooth approximation to `relu`?

## TODO: setup a neural network with different activations







# ## 4.2 Multiple layer neural networks

# An effective NN will have more than one layer. A simple example is if we want to go beyond
# convex functions. Rather than simply summing over the NN we can allow different weights,
# giving us the model
# $$
#   𝐜^⊤ {\rm relu}.(𝐚x + 𝐛) + d.
# $$
# Or we can think of $C = 𝐜^⊤$ as a $1 × n$ matrix. This is in fact a composition of two simple layers, the first being
# $$
#  x ↦ {\rm relu}.(𝐚x + 𝐛)
# $$
# and the second being one without an activation function:
# $$
#  𝐱 ↦ C 𝐱 + d.
# $$
# I.e., they are both `Dense` layers just with different dimensions and different activation functions (`relu` and `identity`).
# We can create such a composition using the `Chain` command:


n = 100
model = Chain(Dense(1 => n, relu), Dense(n => 1))


# Here the parameters are nested. For example, we can create the relevant parameters as follows:


𝐚,𝐛 = randn(n,1),randn(n)
𝐜,d = randn(n),randn(1)

st = (layer_1 = NOSTATE, layer_2 = NOSTATE) # each layer has its own state
ps = (layer_1 = (weight = 𝐚, bias = 𝐛), layer_2 = (weight = 𝐜', bias = d))

@test model([0.1], ps, st)[1] ≈ 𝐜'*relu.(𝐚*0.1 + 𝐛) + d


# We can plot the model evaluated at the gri to see that it is indeed (probably) no longer convex:


plot(x, vec(model(x', ps, st)[1]))


# We now choose the parameters to fit data. Let's generate data for a non-convex function:


x = range(-1, 1; length = n)
y = sin.(3x).*exp.(x)
plot(x,y)


# We will fit this data by minimising the 2-norm with a different model:


function regression_loss(ps, (model, st, (x,y)))
    ỹ = vec(model(x', ps, st)[1])
    norm(ỹ - y) # 2-norm error
end

ps,st = Lux.setup(rng, model)
prob = OptimizationProblem(OptimizationFunction(regression_loss, Optimization.AutoZygote()), ComponentArray(ps), (model, st, (x, y)))
@time ret = solve(prob, Adam(0.03), maxiters=250)

plot(x,y)
plot!(x, vec(model(x', ret.u, st)[1]))



#  It does OK but is still not particularly impressive. The real power in neural networks is their approximation power
# increases as we add more layers. Here let's try an example with 3-layers.


model = Chain(Dense(1 => 200, relu), Dense(200 => 200, relu), Dense(200 => 1))

ps,st = Lux.setup(rng, model)
prob = OptimizationProblem(OptimizationFunction(regression_loss, Optimization.AutoZygote()), ComponentArray(ps), (model, st, (x, y)))
@time ret = solve(prob, Adam(0.03), maxiters=1000)

plot(x,y)
plot!(x, vec(model(x', ret.u, st)[1]))



# **Problem 3** Add a 4th layer and 5th layer, but not all involving square matrices. 
# Can you choose the size of the layers and the activation functions to
# match the eyeball norm? Hint: the answer might be "no" 😅 But maybe "ballpark norm" is sufficient.
## TODO: Setup a NN with 4 and 5 layers.
