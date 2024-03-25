# # SciML SANUM2024
# # Lab 2: Dual Numbers and ForwardDiff.jl

# In this lab we explore a simple approach to computing derivatives:
# _dual numbers_. This is a special mathematical object akin to complex numbers
# that allows us to compute derivatives to very high accuracy in an automated fashion,
# and is an example of forward mode [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).
# To realise dual numbers on a computer we need to introduce the notation of a "type"
# and create a customised type to represent dual numbers, which is what we discuss first.
# For functions of multiple variables we can extend the concept of dual numbers to computing gradients
# and Jacobians.
# After developing our own implementation of dual numbers we investigate using the more sophisticated version
# underlying ForwardDiff.jl. 


# **Learning Outcomes**
#
# 1. Definition and implementation of dual numbers and functions applied dual numbers.
# 2. Using automatic differentiation to implement Newton's method.
# 3. Extending dual numbers to gradients of 2D functions.
# 3. Computing higher-dimensional gradients using ForwardDiff.jl.


# ## 2.1 Dual numbers

# We now consider implementing a type `Dual` to represent the dual number $a + bϵ$,
# in a way similar to `Complex` or `Rat`. For simplicity we don't restrict the types of `a` and `b`
# but for us they will usually be `Float64`. We create this type very similar to `Rat` above:

struct Dual
    a
    b
end

# We can easily support addition of dual numbers as in `Rat` using the formula
# $$
# (a+bϵ) + (c+dϵ) = (a+c) + (b+d)ϵ
# $$

import Base: + # we want to overload +

function +(x::Dual, y::Dual)
    a,b = x.a, x.b # x == a+bϵ. This gets out a and b
    c,d = y.a, y.b # y == c+dϵ. This gets out c and d
    Dual(a+c, b+d)
end

Dual(1,2) + Dual(3,4) # just adds each argument

# For multiplication we used the fact that $ϵ^2 = 0$ to derive the formula
# $$
# (a+bϵ)*(c+dϵ) = ac +(bc+ad)ϵ.
# $$
# Here we support this operation by overloading `*` when the inputs are both
# `Dual`:

import Base: * # we want to also overload *

function *(x::Dual, y::Dual)
    a,b = x.a, x.b # x == a+bϵ. This gets out a and b
    c,d = y.a, y.b # y == c+dϵ. This gets out c and d
    Dual(a*c, b*c + a*d)
end


# ### Differentiating polynomials

# Dual numbers allow us to differentiate functions provided they are composed of
# operations overloaded for `Dual`. In particular, the properties of multiplication imply that
# $$
# (a + b ϵ)^k = a^k + k b a^{k-1} ϵ
# $$
# and therefore by linearity if $f$ is a polynomial it must satisfy
# $$
# f(x + b ϵ) = f(x) + bf'(x)ϵ
# $$
# and thus if we set `b = 1` the "dual part" is equal to the derivative.
# We can use this fact to differentiate simple polynomials that only use `+`
# and `*`:

f = x -> x*x*x + x
f(Dual(2,1)) # (2^3 + 2) + (3*2^2+1)*ϵ

# A polynomial like `x^3 + 1` is not yet supported.
# To support this we need to add addition of `Dual` with `Int` or `Float64`.
# Note that both of these are "subtypes" of `Real` and so restricting on `Real`
# will support both at the same time.
# We can overload the appropriate functions as follows:

import Base: ^

Dual(a::Real) = Dual(a, 0) # converts a real number to a dual number with no ϵ

+(x::Real, y::Dual) = Dual(x) + y
+(x::Dual, y::Real) = x + Dual(y)

## a simple recursive function to support x^2, x^3, etc.
function ^(x::Dual, n::Int)
    if n < 0
        error("Not implemented") # don't support negative n, yet
    end
    if n == 1
        x # Just return the input
    else
        ret = x
        for k = 1:n-1
            ret = ret*x
        end
        ret # returns the last argument
    end
end

f = x -> x^3 + 1
f(Dual(2,1))  # 2^3+1 + 3*2^2*ϵ

# ### Differentiating functions

# We can also overload functions like `exp` so that they satisfy the rules of
# a _dual extension_, that is, are consistent with the formula $f(a+bϵ) = f(a) + bf'(a)ϵ$
# as follows:

import Base: exp
exp(x::Dual) = Dual(exp(x.a), exp(x.a) * x.b)


# We can use this to differentiate a function that composes these basic operations:

f = x -> exp(x^2 + exp(x))
f(Dual(1, 1))


# What makes dual numbers so effective is that, unlike other methods for approximating derivatives like divided differences, they are not
# prone to disasterous growth due to round-off errors: the above approximation
# matches the true answer to roughly 16 digits of accuracy.


# ------

# **Problem 1(a)** Add support for `-`, `cos`, `sin`, and `/` to the type `Dual`
# by replacing the `# TODO`s in the below code.


import Base: -, cos, sin, /

## The following supports negation -(a+bϵ)
-(x::Dual) = Dual(-x.a, -x.b)

## TODO: implement -(::Dual, ::Dual)



function cos(x::Dual)
    ## TODO: implement cos for Duals
    
end

function sin(x::Dual)
    ## TODO: implement sin for Duals
    
end

function /(x::Dual, y::Dual)
    ## TODO: implement division for Duals.
    ## Hint: think of this as x * (1/y)
    
end

x = 0.1
ϵ = Dual(0,1)
@test cos(sin(x+ϵ)/(x+ϵ)).b ≈ -((cos(x)/x - sin(x)/x^2)sin(sin(x)/x))


# **Problem 1(b)** Use dual numbers to compute the derivatives to
# 1. $\exp(\exp x \cos x + \sin x)$
# 2. $∏_{k=1}^{1000} \left({x \over k}-1\right)$
# 3. $f^{\rm s}_{1000}(x)$ where $f^{\rm s}_n(x)$ corresponds to $n$-terms of the following continued fraction:
# $$
# 1 + {x-1 \over 2 + {x-1 \over 2 + {x-1 \over 2 + ⋱}}}.
# $$
# at the point $x = 0.1$. Compare with divided differences to give evidence that your implementation is correct.

## TODO: Use dual numbers to compute the derivatives of the 3 functions above.




# -------

# ## 2.2 Gradients


# Dual numbers extend naturally to higher dimensions by adding a different dual-part for each direction. 
# We will consider a 2D version of a dual number:
# $$
# a + b ϵ_x + c ϵ_y
# $$
# such that
# $$
# ϵ_x^2 = ϵ_y^2 = ϵ_x ϵ_y =  0.
# $$
# Multiplication then follows the rule:
# $$
# (a + b ϵ_x + c ϵ_y) (α + β ϵ_x + γ ϵ_y) = aα + (bα + a β)ϵ_x + (cα + a γ)ϵ_y
# $$
# From this we see
# $$
# \begin{align*}
#  (a + b ϵ_x + c ϵ_y)^k (α + β ϵ_x + γ ϵ_y)^j &= (a^k + k b a^{k-1} ϵ_x + k c a^{k-1} ϵ_y)(α^j + j β α^{j-1} ϵ_x + j γ α^{j-1} ϵ_y) \\
#    &= a^k α^j + (jβ  a^k α^{j-1} + k b a^{k-1} α^j )ϵ_x + (jγ  a^k α^{j-1} + k c a^{k-1} α^j )ϵ_y
# \end{align*}
# $$
# In particular, we have:
# $$
# (x + ϵ_x)^k (y + ϵ_y)^j = x^k y^j + k x^{k-1} y^j ϵ_x + j x^k y^{j-1} ϵ_y
# $$
# and hence by linearity if $f$ is a polynomial we can compute the gradient via:
# $$
# f(x  + ϵ_x, y  + ϵ_y) = f(x,y) + f_x(x,y) ϵ_x + f_y(x,y) ϵ_y.
# $$

# -------

# **Problem 2** 
# Complete the following implementation of `Dual2D` supporting `+` and `*` (and
# assuming `a,b,c` are `Float64`).

import Base: *, +, ^
struct Dual2D
    a::Float64
    b::Float64
    c::Float64
end


function +(s::Dual2D, t::Dual2D)
    ## TODO: Implement +, returning a Dual2D
    
end

function *(c::Number, s::Dual2D)
    ## TODO: Implement c * Dual2D(...), returning a Dual2D
    
end

function *(s::Dual2D, t::Dual2D)
    ## TODO: Implement Dual2D(...) * Dual2D(...), returning a Dual2D
    
    
end

f = function (x, y) # (x+2y^2)^3 using only * and +
    z = (x + 2y * y)
    z * z * z
end

x,y = 1., 2.
@test f(Dual2D(x,1.,0.), Dual2D(y,0.,1.)) == Dual2D(f(x,y), 3(x+2y^2)^2, 12y*(x+2y^2)^2)

## This has computed the gradient as f(x,y) + f_x*ϵ_x + f_y*ϵ_y
## == (x+2y^2)^3 + 3(x+2y^2)^2*ϵ_x + 12y(x+2y^2)^2*ϵ_y

# ----

# ## 2.3 ForwardDiff.jl and computing derivatives/gradients/Jacobians/Hessians

# ForwardDiff.jl is a package that uses dual numbers under the hood for automatic differentiation,
# including supporting gradients and Jacobians. Its usage in 1D works as follows:

using ForwardDiff, Test

@test ForwardDiff.derivative(cos, 0.1) ≈ -sin(0.1) # uses dual number

# It also works with higher dimensions,  allowing for arbitrary dimensional computation
# of gradients. Consider a simple function $f : ℝ^n → ℝ$ defined by
# $$
# f([x_1,…,x_n]) = ∑_{k=1}^{n-1} x_k x_{k+1}
# $$
# which we can implement as follows:

f = function(x)
    ret = zero(eltype(x)) # Need to use zero(eltype(x)) to support dual numbers
    for k = 1:length(x)-1
        ret += x[k]*x[k+1]
    end
    ret
end

# We can use ForwardDiff.jl to compute its gradient:

x = randn(5)
ForwardDiff.gradient(f,x)

# The one catch is the complexity is quadratic:

@time ForwardDiff.gradient(f,randn(1000));
@time ForwardDiff.gradient(f,randn(10_000)); # around 100x slower

# The reason for this is if we have $n$ unknowns the higher-dimensional dual number uses $n$ different $ϵ$s
# for each argument, meaning the input has $n^2$ degrees-of-freedom. 
# This will motivate the move to reverse-mode automatic differentiation in the next lab which will reduce the
# complexity to $O(n)$ for many gradient calculations.

# ### Jacobians
#
# ForwardDiff.jl also works well with Jacobians, a problem where the benefits of reverse-mode automatic differentiation
# are less clear. 
# Denote the Jacobian as
# $$
#  J_f = \begin{bmatrix} {∂ f_1 \over ∂x_1} & ⋯ & {∂ f_1 \over ∂x_ℓ} \\
#       ⋮ & ⋱ & ⋮ \\
#       {∂ f_m \over ∂x_1} & ⋯ & {∂ f_m \over ∂x_ℓ} 
# \end{bmatrix}
# $$
# The function `ForwardDiff.jacobian(f, 𝐱)` computes $J_f(𝐱)$.
# Here is an example of computing the Jacobian of a simple function $f : ℝ^2 → ℝ^2$:

f = function(𝐱)
    (x,y) = 𝐱 # get out the components of the vector
    [exp(x*cos(y)), sin(exp(x*y))]
end

x,y = 0.1,0.2
@test ForwardDiff.jacobian(f, [x,y]) ≈ [exp(x*cos(y))*cos(y)        -exp(x*cos(y))*x*sin(y);
                                        cos(exp(x*y))*exp(x*y)*y     cos(exp(x*y))*exp(x*y)*x]


# -----

# **Problem 3** We can also use ForwardDiff.jl to compute hessians via `ForwardDiff.hessian`. Compute the Hessian of the following Hamiltonian
# $$
#   f([x_1, …, x_n, y_1, …, y_n]) =  {1 \over 2} ∑_{k=1}^n y_k^2 + ∑_{k=1}^{n-1} \exp(x_k - x_{k+1})
# $$

function todahamiltonian(𝐱𝐲)
    n = length(𝐱𝐲) ÷ 2
    x,y = 𝐱𝐲[1:n], 𝐱𝐲[n+1:end] # split the input vector into its two components.
    ret = zero(eltype(𝐱𝐲))
    ## TODO: implement the Hamiltonian, eg using for-loops
    
end

x = [1.,2,3]
y = [4.,5,6]

ForwardDiff.hessian(todahamiltonian, [x; y])

# ----

# ## 2.4 Newton's method

# We will conclude with an application of these results to Newton's method.
# Given an initial guess $x_0$ to a root of a function $f$,  Newton's method is a simple sequence defined by
# $$
# x_{k+1} = x_k - {f(x_k) \over f'(x_k)}
# $$
# If the initial guess $x_0$ is "close enough" to a root $r$ of $f$ (ie $f(r) = 0$)
# then it is known that $x_k → r$. Thus for large $N$ we have $x_N ≈ r$. 

# Dual numbers as implemented by `Dual` gives us a powerful tool to compute derivatives and get a simple implementation
# of Newton's method working:


## derivative(f, x) computes the derivative at a point x using our version of Dual
derivative(f, x) = f(Dual(x,1)).b

function newton(f, x, N) # x = x_0 is the initial guess
    for k = 1:N
        x = x - f(x)/derivative(f,x)
    end
    x
end

f = x -> x^5 + x^2 + 1
r = newton(f, 0.1, 100)

# We can test that we have indeed found a root:
f(r)


# -----


# **Problem 4(a)** Use `newton` with a complex number to compute
# an approximation to a complex root of $f(x) = x^5 - x^2 + 1$.
# Verify the approximation is accurate by testing that it satisfies $f(r)$
# is approximately zero.


## TODO: By making the initial guess complex find a complex root.


# **Problem 4(b)** By changing the initial guesses compute 5 roots to
# $sin(x) - 1/x$. Hint: you may need to add an overload for `/(x::Real, y::Dual)`.

## TODO: Use `newton` to compute roots of `sin(x) - 1/x`.



# **Problem 5** Newton's method works also for finding roots of functions $f : ℝ^n → ℝ^n$ using the Jacobian. 
# Extend our newton method for vector-valued functions:

function newton(f, x::AbstractVector, N) # x = x_0 is the inital guess, now a vector
    ## TODO: reimplement newton for vector inputs using ForwardDiff.jacobian
    
end

f = function(𝐱)
    (x,y) = 𝐱 # get out the components of the vector
    [cos(7x^2*y + y), cos(7*x*y)]
end

@test maximum(abs,f(newton(f, [0.1,0.2], 200))) ≤ 1E-13