# # SciML SANUM2024
# # Lab 3: Reverse-mode automatic differentiation and Zygote.jl

# When the number of unknowns becomes large forward-mode automatic differentiation as
# implemented in ForwardDiff.jl becomes prohibitively expensive for computing gradients and instead we need to
# use reverse-mode automatic differentiation: this is best thought of as implementing the chain-rule
# in an automatic fashion, with a specific choice of multiplying the underlying Jacobians.

# Computing gradients is important for solving optimisation problems, which is what ultimately what training a neural network
# is. Therefore we also look at solving some
# simple optimissation problems, using Optimsation.jl


# **Learning Outcomes**
#
# 1. Computing gradients and derivatives with Zygote.jl
# 2. Basics of reverse-mode automatic differentiation and pullbacks.
# 3. Forward-mode automatic differentiation via pushforwards.
# 4. Using automatic differentiation for implementing gradient descent.
# 5. Solving optimisation with gradient descent and via Optimsation.jl

# ## 3.1 Using Zygote.jl for differentiation

# We begin with a simple demonstration of Zygote.jl, which can be thought of as a replacement for ForwardDiff.jl that
# uses reverse-mode differentiation under the hood. We can differentiate scalar functions, but unlike ForwardDiff.jl it
# overloads the `'` syntax to mean differentiation:


##

# The real power of Zygote.jl is computing gradients (or more generally, Jacobians
# of $f : ℝ^m → ℝ^n$ where $n ≪ m$). We can compute a gradient of the function we considered before as follows:

f = function(x)
    ret = zero(eltype(x))
    for k = 1:length(x)-1
        ret += x[k]*x[k+1]
    end
    ret
end

##


# Unlike ForwardDiff.jl, the gradient returns a tuple since multiple arguments are supported in addition
# to vector inputs, eg:


##


# Now differentiating this function is not particularly faster than ForwardDiff.jl:

##

# It also uses more memory the larger the computation. Take for example 
# the Taylor series for the exponential from Lab 1:

function exp_t(z, n)
    ret = 1.0
    s = 1.0
    for k = 1:n
        s = s/k * z
        ret = ret + s
    end
    ret
end

# The more terms we take the more memory is used, despite the function itself
# using no memory:

##


# Another catch is Zygote.jl doesn't support functions that mutate arrays. Here's an example:

##

# This is unlike `ForwardDiff.gradient` which works fine for differentiating `f!`.


# So why do we use reverse-mode automatic differentiation when it has so many weaknesses
# compared to forward-mode? 
# Because if we write code in just the right way it becomes extremely fast. 
# For example, if we rewrite `f` in a vectorised form we see a huge improvement over
# ForwardDiff.jl:

##


# **Conclusion**: Zygote.jl is much more brittle, sometimes fails outright, requires
# writing functions in a specific way, uses a lot more memory to record complicated operations, but when it works
# well it is _extremely_ fast. Thus when we get to neural networks it is paramount that
# we design our representations of neural networks in a way that is ameniable to reverse-mode
# automatic differentiation, as implemented in Zygote.jl.

# ------



# ## 3.2 Pullbacks and back-propagation for scalar functions

# We now peek a little under-the-hood to get some intuition on how Zygote.jl is computing 
# derivatives, and to understand why its so much faster than ForwardDiff.jl in certain situations. Underlying automatic 
# differentiation in Zygote.jl are so-called "pullback"s. In the scalar
# case these are very close to the notion of a derivative. However, rather than
# the derivative being a single constant, it's a linear map representing the derivative:
# eg, if the derivative of $f(x)$ is denoted $f'(x)$ then the pullback is a linear map
# $$
# t ↦ f'(x)t.
# $$
# We can compute pullbacks using the `pullback` routine:

##

# `p_sin` contains the map $t ↦ \cos(0.1) t$. Since pullbacks support multiple arguments
# it actually returns a tuple with a single entry:

##

# Thus to get out the value we use the following:

##

# The reason its a map instead of just a scalar becomes important for the vector-valued case
# where Jacobians can often be applied to vectors much faster than constructing the Jacobian matrix and
# performing a matrix-vector multiplication.


# Pullbacks can be used for determining more complicated derivatives. Consider a composition of three functions
# $h ∘ g ∘ f$ where from the Chain Rule we know:
# $$
# {{\rm d} \over {\rm d} x}[f(g(h(x))] = f'(g(h(x)) g'(h(x)) h'(x)
# $$
# Essentially we have three pullbacks: the first is the pullback of $f$ evaluated
# at $x$, the second corresponding to $g$ evaluated at $f(x)$, and the third 
# corresponding to $h$ evaluated at $g(f(x))$, that is:
# $$
# \begin{align*}
#  p_1(t) &= f'(x) t  \\
#  p_2(t) &= g'(f(x)) t  \\
#  p_3(t) &= h'(g(f(x))t
# \end{align*}
# $$
# Thus the derivative is given by either the _forward_ or _reverse_ composition of these functions:
# $$
#  p_3(p_2(p_1(1))) = p_1(p_2(p_3(1))) = h'(g(f(x))g'(f(x))f'(x).
# $$
# The first version is called _forward-propagation_ and the second called _back-propagation_.
# Forward-propagation is a version of forward-mode automatic differentiation and is essentially equivalent to using dual numbers.
# We will see later in the vector case that forward- and back-propagation are not the same,
# and that back-propagation is much more efficient provided the output is scalar (or small dimensional).

# Let's see pullbacks in action for computing the derivative of $\cos\sqrt{{\rm e}^x}$:

##

# We can see how this can lead to an approach for automatic differentiation.
# For example, consider the following function composing `sin` over and over:

function manysin(n, x)
    r = x
    for k = 1:n
        r = sin(r)
    end
    r
end

# Now, we would need `n` pullbacks as each time `sin` is called at a different value.
# But the number of such pullbacks grows only linearly so this is acceptable. So thus
# at a high-level we can think of Zygote as running through and computing all the pullbacks:

##

# To deduce the derivative we need can either do forward- or back-propogation by looping through our pullbacks
# either in forward- or in reverse-order. Here we implement back-propagation:

##

# Zygote constructs code that is equivalent to this loop automatically, 
# constructing a high-performance version of this back-propogation loop at compile time using something called source-to-source
# differentiation. But there's no getting around the fact that it needs to record the pullbacks so it does use more memory the larger
# the computation:

##

# ------

# **Problem 1** Compute the derivative of `manysin` using forward-propagation, by looping through the pull-backs
# in the forward direction.

## TODO: loop through pullbacks in order to compute the derivative.



# ## 3.3 Pullbacks with multiple arguments

# Things become more complicated when we have a function with multiple arguments, even in the
# scalar case. Consider now the function $f(g(x), h(x))$. The chain rule tells us that
# $$
# {{\rm d} \over {\rm d} x}[f(g(x), h(x))] = f_x(g(x), h(x)) g'(x) + f_y(g(x), h(x)) h'(x)
# $$
# Now we have three pullbacks:
# $$
# \begin{align*}
# p_1(t) &= g'(x) t\\
# p_2(t) &= h'(x) t\\
# p_3(t) &= [f_x(g(x), h(x))t, f_y(g(x), h(x))t]
# \end{align*}
# $$
# In this case the derivative can be recovered via back-propagation via:
# $$
# p_1(p_3(1)[1]) + p_2(p_3(1)[2]).
# $$
# Here we see a simple example:

##


# Doing more complicated calculations or indeed algorithms becomes
# quite complicated if there are interdependencecies, eg, $f(g(r(x)), h(r(x)))$. 
# This explains why our first version of a function summing over products of its arguments
# was so slow.
# Fortunately, there is an alternative: we can focus on composing vector functions.
# Eg, such a function can be thought of as composition:
# $$
# f ∘ 𝐠 ∘ r
# $$
# where $𝐠(x) = [g(x),h(x)]$. This is a special case of what we discuss in the next section.


# ------

# ## 3.4 Gradients and pullbacks
#
# Now we consider computing gradients of functions that are compositions
# of vector functions, which neural networks fall into.   
# Again, we denote the Jacobian as
# $$
#  J_f = \begin{bmatrix} {∂ f_1 \over ∂x_1} & ⋯ & {∂ f_1 \over ∂x_ℓ} \\
#       ⋮ & ⋱ & ⋮ \\
#       {∂ f_m \over ∂x_1} & ⋯ & {∂ f_m \over ∂x_ℓ} 
# \end{bmatrix}
# $$
# Note that gradients are the transpose of Jacobians: $∇h = J_h^⊤$. 
# For a scalar-valued function $f : ℝ^n → ℝ$ the pullback represents the linear map 
# $p_{f,𝐱} : ℝ → ℝ^n$ corresponding to scaling the gradient:
# $$
# p_{f,𝐱}(t) = J_f(𝐱)^⊤t = ∇f(𝐱) t
# $$
# Here we see an example:

##

# For a function $f : ℝ^n → ℝ^m$ the the pullback represents the linear map $p_{f,𝐱} : ℝ^m → ℝ^n$ given by
# $$
# p_{f,𝐱}(t) = J_f(𝐱)^⊤𝐭
# $$
# Here is a simple example:

##


# Consider a composition $f : ℝ^n → ℝ^m$, $g : ℝ^m → ℝ^ℓ$ and $h : ℝ^ℓ → ℝ$, that is, 
# we want to compute the gradient of $h ∘ g ∘ f : ℝ^n → ℝ$. The Chain rule tells us that
# $$
#  J_{h ∘ g ∘ f}(𝐱) = J_h(g(f(𝐱)) J_g(f(𝐱)) J_f(𝐱)
# $$
# Put another way, the gradiant of $h ∘ g ∘ f$
# is given by the transposes of Jacobians:
# $$
#    ∇[{h ∘ g ∘ f}](𝐱) = J_f(𝐱)^⊤ J_g(f(𝐱))^⊤  ∇h(g(f(𝐱))
# $$
# Thus we have three pullbacks $p_1 : ℝ^m → ℝ^n$, $p_2 : ℝ^ℓ → ℝ^m$ and $p_3 : ℝ → ℝ^ℓ$ given by
# \begin{align*}
#  p_1(𝐭) &= J_f(𝐱)^⊤ 𝐭  \\
#  p_2(𝐭) &= J_g(f(x))^⊤ 𝐭  \\
#  p_3(t) &= ∇h(g(f(𝐱)) t
# \end{align*}
# The gradient is given by _back-propagation_:
# $$
#  p_1(p_2(p_3(1))) = J_f(𝐱)^⊤ J_g(f(𝐱))^⊤  ∇h(g(f(𝐱)).
# $$
# Here the "right" order to do the multiplications is clear: matrix-matrix multiplications are expensive
# so its best to do it reverse order so that we only ever have matrix-vector multiplications.
# Also, the pullback doesn't give us enough information to implement forward-propagation:
# we don't have access to the Jacobian matrices, or their application.

# As an example consider computing the gradient of an iteration a simple map like:
# $$
# 𝐟(x,y,z) = \begin{bmatrix} \cos(xy)+z\\ zy-\exp(x)\\ x + y + z \end{bmatrix}
# $$
# and summing over the result, eg. computing $[1,1,1]^⊤(\underbrace{𝐟 ∘ ⋯ ∘ 𝐟}_{n\hbox{ times}})(𝐱)$.
# We implement this with a general function `iteratef`:
##

# To get an idea how this works behind the scenes we can again accumulate the pullbacks:

##

# We can recover the gradient by back-propogation:

##


# Indeed we match the gradient as computed with Zygote.jl:

##


# **Problem 2** The function `pushforward` represent the map $𝐭 ↦ J_f(𝐱) 𝐭$. 
# Compute the gradient of `iteratef` as above with forward-mode automatic differentiation by using `pushforward`.
# Do so without creating a vector of pushforwards.
# Hint: We need to run the pushforward iteration with the identity matrix as the initial value,
# but the result of  `pushforward` only works on vectors. So we need to apply it to each column of the matrix manually.

## TODO: Compute the gradient as above but using pushforward


# **Problem 3** Consider a simple forward Euler method approximating the solution to the Pendulum equation with friction:
# $$
# u'' = τ u' - \sin u
# $$
# which we can rewrite as a first order system:
# $$
# \begin{bmatrix}
#    u' \\
#    v'
#    \end{bmatrix} = \begin{bmatrix} v \\ -τ*v - \sin u \end{bmatrix}
# $$
# That is, we want to implement the iteration
# $$
# 𝐮_{k+1} = 𝐮_k + h*\begin{bmatrix} 𝐮_k[2] \\ -τ 𝐮_k[2] - \sin 𝐮_k[1] \end{bmatrix}
# $$
# with a specified initial condition $𝐮_0$. For $N = 100$, $h = 0.1$ and $𝐮_0 = [0.1,0.2]$, differentiate
# the solution with-respect to $τ$ at $τ = 1$ by creating a vector of pullbacks and implementing back-propagation.
# Hint: Forward Euler is a variant of `iteratef` above so you can modify the subsequent pullback construction. Add $τ$ to the vector
# of values to capture the relevant dependencies and verify your result by comparing to `gradient`.






# ## 3.5 Optimisation

# A key place where reverse-mode automatic differentiation is essential is large scale optimisation.
# As a  simple example we will look at the classic optimisation problem
# that solves $A 𝐱 = 𝐛$ where $A$ is symmetric positive definite: find $𝐱$ that minimises
# $$
# f_{A,𝐛}(𝐱) = 𝐱^⊤ A 𝐱 - 2𝐱^⊤ 𝐛.
# $$.
# Of course we can use tried-and-true techniques implemented in `\` but here we want
# to emphasise we can also solve this with simple optimsation algorithms like gradient desecent
# which do not know the structure of the problem. We consider a matrix where we know gradient descent
# will converge fast:
# $$
# A = \begin{bmatrix} 1 & 1/2^α \\ 1/2^α & 1 & ⋱ \\ &  ⋱ & ⋱ & 1/n^α \\ && 1/n^α & 1 \end{bmatrix}
# $$
# In other words we want to minimise the functional (or the _loss function_)
# $$
# f_{A,𝐛}(𝐱) = ∑_{k=1}^n x_k^2 + ∑_{k=2}^n x_{k-1} x_k/k^α - ∑_{k=1}^n x_k b_k.
# $$
# For simplicity we will take $𝐛$ to be the vector with all ones.

# Owing to the constraints of Zygote.jl, we need to write this in a vectorised way to ensure Zygote is sufficiently fast. 
# Here we see that when we do this we can efficiently
# compute gradients even
# with a million degrees of freedom, way beyond what could ever be done with forward-mode automatic differentiation:

##

# For concreteness we first implement our own version of a quick-and-dirty gradient descent:
# $$
# x_{k+1} = x_k - γ_k ∇f(x_k)
# $$
# where $γ_k$ is the learning rate. To choose $γ_k$ we just halve
# the learning rate until we see decrease in the loss function.

##


# We can compare this with the "true" solution:

##

# In practice its better to use inbuilt optimsation routines and packages. Here we see how we can solve the same problem with
# the Optimization.jl package, combined with OptimizationOptimisers.jl that has gradient-based optimisation methods,
# in particular `Adam`.

##



# **Problem 4** This problem considers an example that will connect with  neural networks.
# Define ${\rm relu}(x) := \max(0,x)$ and consider an approximation of the form:
# $$
# p_{𝐚,𝐛}(x) := ∑_{k=1}^n {\rm relu}(a_k x + b_k)
# $$
# where $𝐚,𝐛 ∈ ℝ^n$. This is a sum of positive convex functions hence consider regression for a positive convex function
# like $f(x) =  \exp x$. For $n = 100$,  approximate $𝐚,𝐛$ that minimises $\|p_{𝐚,𝐛}.(𝐱) - f.(𝐱)\|$ where $𝐱$ is a vector containing
# 100 evenly spaced points between $-1$ and $1$ (inclusive). Compare your regression with $f$ by plotting the two functions.


## TODO: Construct a model for the function and perform regression using Optimization.jl
