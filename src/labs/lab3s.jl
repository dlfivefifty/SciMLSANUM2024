# # SciML SANUM2024
# # Lab 3: Reverse-mode automatic differentiation and Zygote.jl

# When the number of unknowns becomes large forward-mode automatic differentiation as
# implemented in ForwardDiff.jl becomes prohibitively expensive and instead we need to
# use reverse-mode automatic differentiation: this is best thought of as implementing the chain-rule
# in an automatic fashion. 


# **Learning Outcomes**
#
# 1. Basics of reverse-mode automatic differentiation and pullbacks.
# 2. Implementation via Zygote.jl
# 3. Adding custom pullbacks.
# 4. Using automatic differentiation for implementing gradient descent.


using Zygote

f = function(x)
    ret = zero(eltype(x))
    for k = 1:length(x)-1
        ret += x[k]*x[k+1]
    end
    ret
end

f = x -> sum(x[1:end-1] .* x[2:end])

f = x -> sum(sin, x)

x = randn(5)
Zygote.gradient(f,x)

# The one catch is the complexity is quadratic:
Base.GC.enable(false)
n = 100_000_000; @time Zygote.gradient(f,randn(n));

Base.GC.enable(true)
Base.GC.gc()
@time Zygote.gradient(f,randn(10_000)); # around 100x slower

using ForwardDiff
@time ForwardDiff.gradient(f,randn(10_000)); # around 100x slower

# Under the scenes this determines so-called "pullback"s. In the scalar
# case these are very close to the notion of a derivative. However, rather than
# the derivative being a single constant, its a linear map: eg, if the derivative
# of $f(x)$ is denoted $f'(x)$ then the pullback is a linear map
# $$
# t ↦ tf'(x).
# $$
# In the scalar case this can also be written in a more natural $f'(x)t$ form
# but the order will become important when we discuss non-scalar functions. 
# We can compute pullbacks using the `pullback` routine:

import Zygote: pullback

s, sin_J = pullback(sin, 0.1)

# `sin_J` contains the map $t ↦ t \cos 0.1$. Since pullbacks support multiple arguments
# it actually returns a tuple with a single entry:

sin_J(1)

# Thus to get out the value we use the following:

@test sin_J(1)[1] == cos(0.1)
@test sin_J(2)[1] == 2cos(0.1)

# The reason its a map instead of a function becomes important for composing functions. Lets consider
# a composition of three functions:
# $$
# {\rm d} \over {\rm d x}[f(g(h(x))] = f'(g(h(x)) g'(h(x)) h'(x)
# $$
# Essentially we have three pullbacks: the first is the pullback of $h$ evaluated
# at $x$, the second corresponding to $g$ evaluated at $h(x)$, and the third 
# corresponding to $f$ evaluated at $g(h(x))$, that is:
# $$
# \begin{align*}
#  p_1(t) &= t h'(x) \\
#  p_2(t) &= t g'(h(x)) \\
#  p_3(t) &= t f'(g(h(x))
# \end{align*}
# $$
# Thus the derivative is given by the reverse composition:
# $$
#  p_1(p_2(p_3(1)))
# $$
# In the scalar case, these are linear and commuting maps so you might wonder
# why not use an arguably more natural $p_3(p_2(p_1(1))$, but the reverse order
# becomes essential in higher dimensions.

# Let's see this in action for computing the derivative of $\cos\sqrt{{\rm e}^x}$:

x = 0.1 # point we want to differentiate
y,p₁ = pullback(exp, x) 
z,p₂ = pullback(sqrt, y) # y is exp(x)
w,p₃ = pullback(cos, z) # z is sqrt(exp(x))

@test w == cos(sqrt(exp(x)))

@test p₁(p₂(p₃(1)...)...)[1] ≈ -sin(sqrt(exp(x)))*exp(x)/(2sqrt(exp(x)))


# We can see how this can lead to an approach for automatic differentiation.
# For example,   consider the following function composing `sin`` over and over:

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

# To deduce the derivative we need to do "back-propogation": loop through our pullbacks
# in reverse:

der = 1 # we always initialise with the trivial scaling
for k = n:-1:1
    der = pullbacks[k](der)[1]
end
@test der ≈ ForwardDiff.derivative(x -> manysin(n, x), x)

# Zygote constructs code that is equivalent to this loop automatically, and without the need for creating a "tape",
# that is, it doesn't actually allocate a vector `pullbacks` to record the operations at run-time, rather,
# deduces a high-performance version of this back-propogation loop at compile time.

# **Problem** Write a simple function for Taylor series of exponential. 
# Write a function that implements back-propogation

# ## Gradients
#
# Above we introduced  scalar. Now we consider computing gradients, which is
# essential in ML. Consider a now $f : ℝ^ℓ → ℝ^m$, $g : ℝ^m → ℝ^n$ and $h : ℝ^n → ℝ$.
# Denote the Jacobian as
# $$
#  J_f = \begin{bmatrix} {∂ f_1 \over ∂x_1} & ⋯ & {∂ f_1 \over ∂x_ℓ} \\
#       ⋮ & ⋱ & ⋮ \\
#       {∂ f_m \over ∂x_1} & ⋯ & {∂ f_m \over ∂x_ℓ} 
# \end{bmatrix}
# $$
# The Chain rule tells us that
# $$
#  J_{h ∘ g ∘ f}(𝐱) = J_h(g(f(𝐱)) J_g(f(𝐱)) J_f(𝐱)
# $$
# Note that gradients are the transpose of Jacobians: $∇h = J_h^⊤$. Put another way, the gradiant of $h ∘ g ∘ f$
# is given by the transposes of Jacobians:
# $$
#    ∇[{h ∘ g ∘ f}](𝐱) = J_f(𝐱)^⊤ J_g(f(𝐱))^⊤  ∇h(g(f(𝐱))
# $$
# We can compute this using pullbacks. 

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


# Here the "right" order to do the multiplications is clear: matrix-matrix multiplications are expensive
# so its best to reverse order:



z,pb = rrule(sum, [1, 2])

Zygote.jacobian(x -> broadcast(sin, x), [1,2])[1]
rrule(broadcast, sin, [1,2])

unthunk(pb(1)[2])

@ent Zygote.gradient(sum, [1,2])

