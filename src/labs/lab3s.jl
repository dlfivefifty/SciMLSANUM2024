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
# 1. Basics of reverse-mode automatic differentiation and pullbacks.
# 2. Implementation via Zygote.jl
# 3. Adding custom pullbacks.
# 4. Using automatic differentiation for implementing gradient descent.
# 5. Solving optimisation with gradient descent and via Optimsation.jl

# 3.1 Using Zygote.jl for differentiation

# We begin with a simple demonstration of Zygote.jl, which can be thought of as a replacement for ForwardDiff.jl that
# uses reverse-mode differentiation under the hood. We can differentiate scalar functions, but unlike ForwardDiff.jl it
# overloads the `'` syntax to mean differentiation:


using Zygote, Test

@test cos'(0.1) ≈ -sin(0.1) # Differentiates cos using reverse-mode autodiff

# Gradients can be computed as follows:


f = function(x)
    ret = zero(eltype(x))
    for k = 1:length(x)-1
        ret += x[k]*x[k+1]
    end
    ret
end

x = randn(5)
Zygote.gradient(f,x)

# Unlike ForwardDiff.jl, the gradient returns a tuple since multiple arguments are supported in addition
# to vector inputs, eg:

x,y = 0.1, 0.2
@test all(Zygote.gradient((x,y) -> cos(x*exp(y)), x, y) .≈ [-sin(x*exp(y))*exp(y), -sin(x*exp(y))*x*exp(y)])


# Now differentiating this function is not particularly faster than ForwardDiff.jl:

x = randn(1000)
@time Zygote.gradient(f, x);
x = randn(10_000)
@time Zygote.gradient(f, x); # roughly 200x slower

# This is because not all operations are ameniable to reverse-mode differentiation as implemented in Zygote.jl.
# However, if we restrict to vectorised operations we see a dramatic improvement:

f_vec = x -> sum(x[1:end-1] .* x[2:end]) # a vectorised version of the previus function
Zygote.gradient(f_vec, x) # compile
@time Zygote.gradient(f_vec, x); #  1500x faster 🤩

# Another catch is Zygote.jl doesn't support functions that mutate arrays. Here's an example:

f! = function(x)
    n = length(x)
    ret = zeros(eltype(x), n)
    for k = 1:n-1
        ret[k] = x[k]*x[k+1] # modifies the vector ret
    end
    sum(ret)
end


x = randn(5)
Zygote.gradient(f!,x) # errors out

# This is unlike `ForwardDiff.gradient` which works fine for differentiating `f!`.
# The conclusion: Zygote.jl is much more brittle, sometimes fails outright, but when it works
# well it is _extremely_ fast. Thus when we get to neural networks it is paramount that
# we design our representation of the neural network in a way that is ameniable to reverse-mode
# automatic differentiation.

# ------



# ## 3.2 Pullbacks and back-propagation for scalar functions

# We now peek a little under-the-hood to get some intuition on how Zygote.jl is computing 
# derivatives. Underlying automatic differentiation in Zygote.jl isare so-called "pullback"s. In the scalar
# case these are very close to the notion of a derivative. However, rather than
# the derivative being a single constant, its a linear map: eg, if the derivative
# of $f(x)$ is denoted $f'(x)$ then the pullback is a linear map
# $$
# t ↦ f'(x)t.
# $$
# We can compute pullbacks using the `pullback` routine:

import Zygote: pullback

s, sin_J = pullback(sin, 0.1)

# `sin_J` contains the map $t ↦ \cos(0.1) t$. Since pullbacks support multiple arguments
# it actually returns a tuple with a single entry:

sin_J(1)

# Thus to get out the value we use the following:

@test sin_J(1)[1] == cos(0.1)
@test sin_J(2)[1] == 2cos(0.1)

# The reason its a map instead of just a scalar becomes important for the vector-valued case
# where Jacobians can often be applied to vectors much faster than creating a matrix and
# performing a matrix-vector multiplication.


# Pullbacks can be used for determining more complicated derivatives. Consider a composition of three functions
# $h ∘ g ∘ f$ where from the Chain Rule we know:
# $$
# {\rm d} \over {\rm d x}[f(g(h(x))] = f'(g(h(x)) g'(h(x)) h'(x)
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
# Forward-propagation is a version of forward-mode automatic differentiation and is essentially equivalent to using dual number.
# We will see in the vector case that back-propagation is much more efficient.

# Let's see this in action for computing the derivative of $\cos\sqrt{{\rm e}^x}$:

x = 0.1 # point we want to differentiate
y,p₁ = pullback(exp, x) 
z,p₂ = pullback(sqrt, y) # y is exp(x)
w,p₃ = pullback(cos, z) # z is sqrt(exp(x))

@test w == cos(sqrt(exp(x)))

@test p₁(p₂(p₃(1)...)...)[1] ≈ p₃(p₂(p₁(1)...)...)[1] ≈ -sin(sqrt(exp(x)))*exp(x)/(2sqrt(exp(x)))


# We can see how this can lead to an approach for automatic differentiation.
# For example, consider the following function composing `sin`` over and over:

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

n = 5
x = 0.1 # input

pullbacks = Any[] # a vector where we store the pull backs
r = x
for k = 1:n
    r,pₖ = pullback(sin, r) # new pullback
    push!(pullbacks, pₖ)
end
r # value

# To deduce the derivative we need can either do forward- or back-back-propogation: loop through our pullbacks.

forward_der = 1 # we always initialise with the trivial scaling
for k = 1:n
    forward_der = pullbacks[k](forward_der)[1]
end

reverse_der = 1 # we always initialise with the trivial scaling
for k = n:-1:1
    reverse_der = pullbacks[k](reverse_der)[1]
end
@test reverse_der ≈ forward_der ≈ ForwardDiff.derivative(x -> manysin(n, x), x)

# Zygote constructs code that is equivalent to this loop automatically, and without the need for creating a "tape".
# That is, it doesn't actually create a vector `pullbacks` to record the operations at run-time, rather,
# it constructs a high-performance version of this back-propogation loop at compile time using something called source-to-source
# differentiation.

# ------

# **Problem** Write a simple function for Taylor series of exponential. 
# Write a function that implements back-propogation

# ------

# ## 3.3 Gradients and pullbacks
#
# Now we consider computing gradients, which is
# essential in ML.   Again we denote the Jacobian as
# $$
#  J_f = \begin{bmatrix} {∂ f_1 \over ∂x_1} & ⋯ & {∂ f_1 \over ∂x_ℓ} \\
#       ⋮ & ⋱ & ⋮ \\
#       {∂ f_m \over ∂x_1} & ⋯ & {∂ f_m \over ∂x_ℓ} 
# \end{bmatrix}
# $$
# Note that gradients are the transpose of Jacobians: $∇h = J_h^⊤$. 
# For a scalar-valued function $f : ℝ^n → ℝ$ the pullback represents the linear map $ℝ → ℝ^n$
# corresponding to the gradient:
# $$
# 𝐭 ↦ J_f(𝐱)^⊤𝐭 = ∇f(𝐱) 𝐭
# $$
# Here we see an example:


f = (𝐱) -> ((x,y) = 𝐱;  exp(x*cos(y)))
𝐱 = [0.1,0.2]
f_v, f_pb = Zygote.pullback(f, 𝐱)
@test f_pb(1)[1] ≈ [exp(x*cos(y))*cos(y), -exp(x*cos(y))*x*sin(y)]

# For a function $f : ℝ^n → ℝ^m$ the the pullback represents the map $ℝ^m → ℝ^n$ given by
# $$
# 𝐭 ↦ J_f(𝐱)^⊤𝐭
# $$
# Here is an example:

f = function(𝐱)
    x,y,z = 𝐱
    [exp(x*y*z),cos(x*y+z)]
end
     

𝐱 = [0.1,0.2,0.3]
f_v, f_pb =  pullback(f, 𝐱)

J_f = function(𝐱)
    x,y,z = 𝐱
    [y*z*exp(x*y*z) x*z*exp(x*y*z) x*y*exp(x*y*z);
     -y*sin(x*y+z) -x*sin(x*y+z) -sin(x*y+z)]
end

𝐲 = [1,2]
@test J_f(𝐱)'*𝐲 ≈ f_pb(𝐲)[1]


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
#  p_3(𝐭) &= ∇h(g(f(𝐱)) 𝐭
# \end{align*}
# The gradient is given by _back-propagation_:
# $$
#  p_1(p_2(p_3(1))) = J_f(𝐱)^⊤ J_g(f(𝐱))^⊤  ∇h(g(f(𝐱)).
# $$
# Here the "right" order to do the multiplications is clear: matrix-matrix multiplications are expensive
# so its best to do it reverse order so that we only ever have matrix-vector multiplications.
# Also, the pullback doesn't give us enough information to implement forward-propagation.
# (This can be done using `Zygote.pushforward` which implements the transpose map $𝐭 ↦ J_f(𝐱) 𝐭$
# and is used in Zygote.jl for 



z,pb = rrule(sum, [1, 2])

Zygote.jacobian(x -> broadcast(sin, x), [1,2])[1]
rrule(broadcast, sin, [1,2])

unthunk(pb(1)[2])

@ent Zygote.gradient(sum, [1,2])



# ## 3.4 Optimisation

# A key place where reverse-mode automatic differentiation is large scale optimisation.

# As an extremely simple example we will look at the classic optimisation problem
# representing $A 𝐱 = 𝐛$: find $𝐱$ that minimises
# $$
# f_{A,𝐛}(𝐱) = {1 \over 2} 𝐱^⊤ A 𝐱 - 𝐱^⊤ 𝐛
# $$
# where $A$ is symmetric positive definite.
# Of course we can use tried-and-true techniques like the QR factorisation but here we want
# to emphasise we can also solve this with simple optimsation algorithms like gradient desecent
# which do not know the structure of the problem. For simplicity we will actually consider a square
# problem with a classic finite-difference stencil:
# $$
# A = {n^2} \begin{bmatrix} 2 & -1 \\ -1 & 2 & ⋱ \\ &  ⋱ & ⋱ & -1 \\ && -1 & 2 \end{bmatrix}
# $$
# In other words our functional has the form:
# $$
# f_{A,𝐛}(𝐱) = {n^2}(∑_{k=1}^n x_k^2 - ∑_{k=1}^{n-1} x_k x_{k+1}) - ∑_{k=1}^n x_k b_k
# $$
# We need to write this in a vectorised way to ensure Zygote is sufficiently fast. Here is the code
# with a million degrees of freedom, way beyond what could ever vbe done 

n = 1000

f = x -> (2x'x - 2x[1:end-1]'x[2:end])*n^2 - 2sum(x)

x = randn(n) # initial guess
Zygote.gradient(f, x) # compile
@time Zygote.gradient(f, x)

# Our algorithm is a simple gradient descent:
# $$
# x_{k+1} = x_k - γ ∇f(x_k)
# $$
# where $γ$ is the learning rate. Here's a simple implementation with learning rate equal to 1:

x = randn(n)
for k = 1:10_000
    γ = 1
    y = x - γ*Zygote.gradient(f, x)[1]
    while f(x) < f(y)
        γ /= 2 # half the learning rate
        y = x - γ*Zygote.gradient(f, x)[1]
    end
    x = y
    @show γ,f(x)
end

A = SymTridiagonal(fill(2,n), fill(-1,n-1))*n^2


f(A \ ones(n))
plot(A \ ones(n))

n = 1_00; h = 1/n;

y = ones(n)
f = y -> sum(y[1:end-1] .* sqrt.(1 .+ ((y[2:end] - y[1:end-1])/h) .^2))/n

@time Zygote.gradient(f, y)

for k = 1:1_000
    γ = 1
    z = y - γ*Zygote.gradient(f, y)[1]
    z[1] = z[end] = 1
    while f(y) < f(z)
        γ /= 2 # half the learning rate
        z = y - γ*Zygote.gradient(f, y)[1]
        z[1] = z[end] = 1
    end
    y = z
    @show γ,f(y)
end

plot(y)
# 

using Optimization, OptimizationOptimJL


n = 1000
f = (y,(a,b,h)) -> (sqrt(1 + ((y[1] - a)/h).^2) + sum(y[1:end-1] .* sqrt.(1 .+ ((y[2:end] - y[1:end-1])/h) .^2)) + sqrt(1 + ((b - y[end])/h).^2))*h
y0 = ones(n-2) # drop boundary conditions
prob = OptimizationProblem(OptimizationFunction(f, Optimization.AutoZygote()), y0, (1,1,1/n))
@time y = solve(prob, BFGS()); plot(y)

@time y = solve(prob, GradientDescent()); plot(y)


