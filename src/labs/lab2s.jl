# # SciML SANUM2024
# # Lab 2: Dual Numbers and ForwardDiff.jl

# In this lab we explore a simple approach to computing derivatives:
# _dual numbers_. This is a special mathematical object akin to complex numbers
# that allows us to compute derivatives to very high accuracy in an automated fashion,
# and is an example of forward mode [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).
# To realise dual numbers on a computer we need to introduce the notation of a "type"
# and create a customised type to represent dual numbers, which is what we discuss first.

# After developing our own implementation of dual numbers we investigate using the more sophisticated version
# underlying ForwardDiff.jl. This version allows for computing gradients, which are useful for
# optimisation problems. We will use this for a simple implementation of gradient descent.


# **Learning Outcomes**
#
# 1. Definition and implementation of dual numbers and functions applied dual numbers.
# 2. Using automatic differentiation to implement Newton's method
# 3. Extension to gradients as implemented in ForwardDiff.jl
# 4. Using automatic differentiation for implementing gradient descent.

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


# ### I.3.1 Differentiating polynomials

# Dual numbers allow us to differentiate functions provided they are composed of
# operations overloaded for `Dual`. In particular, we have that
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

# a simple recursive function to support x^2, x^3, etc.
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

# ### I.3.2 Differentiating functions

# We can also overload functions like `exp` so that they satisfy the rules of
# a _dual extension_, that is, are consistent with the formula $f(a+bϵ) = f(a) + bf'(a)ϵ$
# as follows:

import Base: exp
exp(x::Dual) = Dual(exp(x.a), exp(x.a) * x.b)


# We can use this to differentiate a function that composes these basic operations:

f = x -> exp(x^2 + exp(x))
f(Dual(1, 1))


# What makes dual numbers so effective is that, unlike divided differences, they are not
# prone to disasterous growth due to round-off errors: the above approximation
# matches the true answer to roughly 16 digits of accuracy.


# ------

# **Problem 4(a)** Add support for `-`, `cos`, `sin`, and `/` to the type `Dual`
# by replacing the `# TODO`s in the below code.


import Base: -, cos, sin, /

## The following supports negation -(a+bϵ)
-(x::Dual) = Dual(-x.a, -x.b)

## TODO: implement -(::Dual, ::Dual)
## SOLUTION
-(x::Dual, y::Dual) = Dual(x.a - y.a, x.b - y.b)
## END


function cos(x::Dual)
    ## TODO: implement cos for Duals
    ## SOLUTION
    Dual(cos(x.a), -sin(x.a) * x.b)
    ## END
end

function sin(x::Dual)
    ## TODO: implement sin for Duals
    ## SOLUTION
    Dual(sin(x.a), cos(x.a) * x.b)
    ## END
end

function /(x::Dual, y::Dual)
    ## TODO: implement division for Duals.
    ## Hint: think of this as x * (1/y)
    ## SOLUTION
    if iszero(y.a)
        error("Division for dual numbers is ill-defined when denonimator real part is zero.")
    end
    return Dual(x.a / y.a, (y.a * x.b - x.a * y.b) / y.a^2)
    ## END
end

x = 0.1
ϵ = Dual(0,1)
@test cos(sin(x+ϵ)/(x+ϵ)).b ≈ -((cos(x)/x - sin(x)/x^2)sin(sin(x)/x))


# **Problem 4(b)** Use dual numbers to compute the derivatives to
# 1. $\exp(\exp x \cos x + \sin x)$
# 2. $∏_{k=1}^{1000} \left({x \over k}-1\right)$
# 3. $f^{\rm s}_{1000}(x)$ where, as in Lab 1 Problem 3(d), $f^{\rm s}_n(x)$ corresponds to $n$-terms of the following continued fraction:
# $$
# 1 + {x-1 \over 2 + {x-1 \over 2 + {x-1 \over 2 + ⋱}}}.
# $$
# at the point $x = 0.1$. Compare with divided differences to give evidence that your implementation is correct.

## TODO: Use dual numbers to compute the derivatives of the 3 functions above.
## SOLUTION

## Define the functions
f = x -> exp(exp(x)cos(x) + sin(x))
/(x::Dual, k::Int) = Dual(x.a/k, x.b/k) # missing overload from above
-(x::Dual, k::Int) = Dual(x.a-k, x.b) # missing overload from above
*(k::Int, x::Dual) = Dual(k*x.a, k*x.b) # missing overload from above
g = function(x)
    ret = 1
    for k = 1:1000
        ret *= x/k - 1
    end
    ret
end
function cont(n, x)
    ret = 2
    for k = 1:n-1
        ret = 2 + (x-1)/ret
    end
    1 + (x-1)/ret
end

## With the previous problems solved, this is as simple as running

fdual = f(0.1+ϵ)
fdual.b
#
gdual = g(0.1+ϵ)
gdual.b
#
contdual = cont(1000,0.1+ϵ)
contdual.b
## END





# ## Gradients


# **Problem 2.2 (A)** Consider a 2D version of a dual number:
# $$
# a + b ϵ_x + c ϵ_y
# $$
# such that
# $$
# ϵ_x^2 = ϵ_y^2 = ϵ_x ϵ_y =  0.
# $$
# Complete the following implementation supporting `+` and `*` (and
# assuming `a,b,c` are `Float64`). Hint: you may need to work out on paper
# how to multiply `(s.a + s.b ϵ_x + s.c ϵ_y)*(t.a + t.b ϵ_x + t.c ϵ_y)` using the
# relationship above.

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

# This has computed the gradient as f(x,y) + f_x*ϵ_x + f_y*ϵ_y
# == (x+2y^2)^3 + 3(x+2y^2)^2*ϵ_x + 12y(x+2y^2)^2*ϵ_y

# ##

# ForwardDiff.jl is a package that uses dual numbers under the hood for automatic differentiation. 

using ForwardDiff, Test

@test ForwardDiff.derivative(cos, 0.1) ≈ -sin(0.1) # uses dual number

# It also works with higher dimensions,  allowing for arbitrary dimensional computation
# of gradients. Consider a simple function:

f = function(x)
    ret = zero(eltype(x))
    for k = 1:length(x)-1
        ret += x[k]*x[k+1]
    end
    ret
end

f = function(x)
    ret = zero(eltype(x))
    for k = 1:length(x)-1
        ret += x[k]^2
    end
    ret
end

x = randn(5)
ForwardDiff.gradient(f,x)

# The one catch is the complexity is quadratic:

@time ForwardDiff.gradient(f,randn(1000));
@time ForwardDiff.gradient(f,randn(10_000)); # around 100x slower


# This will motivate the move to reverse-mode automatic differentiation.

# ## I.4 Newton's method

# Newton's method is a simple algorithmic approach that you may have seen before in school for computing roots (or zeros)
# of functions. The basic idea is given an initial guess $x_0$,
# find the first-order Taylor approximation $p(x)$ (i.e., find the line that matches the slope of the function at the point)
# $$
# f(x) ≈ \underbrace{f(x_0) + f'(x_0) (x- x_0)}_{p(x)}.
# $$
# We can then solve the root finding problem for $p(x)$ exactly:
# $$
# p(x) = 0 ⇔ x = x_0 - {f(x_0) \over f'(x_0)}
# $$
# We take this root of $p(x)$ as the new initial guess and repeat. In other words, we have a simple sequence
# defined by
# $$
# x_{k+1} = x_k - {f(x_k) \over f'(x_k)}
# $$
# If the initial guess is "close enough" to a root $r$ of $f$ (ie $f(r) = 0$)
# then it is known that $x_k → r$. Thus for large $N$ we have $x_N ≈ r$. Note the notion of "close enough"
# is a complicated and rich theory beyond the scope of this module, and connects to the theory of [Mandelbrot sets](https://en.wikipedia.org/wiki/Mandelbrot_set).

# Dual numbers as implemented by `Dual` gives us a powerful tool to compute derivatives and get a simple implementation
# of Newton's method working:


## derivative(f, x) computes the derivative at a point x using Dual
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

# **Problem 5(a)** For $f(x) = x^5 + x^2 + 1$, plot the error of $x_k$ for `k = 1:15` where the
# y-axis is scaled logarithmically and chosen $x_0 = 0.1$ You may
# use the computed `r` as the "exact" root. What do you think the convergence rate is?

## TODO: compute and plot the error of `newton(f, 0.1, k)` for `k = 1:15`
## SOLUTION
plot(1:15, [nanabs(newton(f, 0.1, k)-r) for k=1:15]; yscale=:log10)
## It converges faster than exponentially.
## END

# **Problem 5(b)** Use `newton` with a complex number to compute
# an approximation to a complex root of $f(x) = x^5 - x^2 + 1$.
# Verify the approximation is accurate by testing that it satisfies $f(r)$
# is approximately zero.


## TODO: By making the initial guess complex find a complex root.
## SOLUTION
r = newton(f, 0.1 + 0.2im, 100)
f(r) # close to zero
## END

# **Problem 5(c)** By changing the initial guesses compute 5 roots to
# $sin(x) - 1/x$. Hint: you may need to add an overload for `/(x::Real, y::Dual)`.

## TODO: Use `newton` to compute roots of `sin(x) - 1/x`.
## SOLUTION

## We need to add a missing overload for `Dual`:

/(x::Real, y::Dual) = Dual(x)/y


## Changing the initial guess we get 5 distinct roots
newton(x -> sin(x) - 1/x, 1, 100),
newton(x -> sin(x) - 1/x, 2, 100),
newton(x -> sin(x) - 1/x, 3, 100),
newton(x -> sin(x) - 1/x, 5, 100),
newton(x -> sin(x) - 1/x, 6, 100)

## END
