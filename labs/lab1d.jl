# # SciML SANUM2024
# # Lab 1: Introduction to Julia


# In this course we will use the programming language Julia. This is a modern, compiled, 
# high-level, open-source language developed at MIT. It is becoming increasingly 
# important in high-performance computing and AI, including by Astrazeneca, Moderna and 
# Pfizer in drug development and clinical trial accelleration, IBM for medical diagnosis, 
# MIT for robot locomotion, and elsewhere.

# It is ideal for a course on Scientific Machine Learning (SciML)  because it both allows fast 
# implementation of algorithms but also has support for fast automatic-differentiation, 
# a feature that is of increasing importance in machine learning. 
# Also, the libraries for solving differential equations and SciML are quite advanced.
# As a bonus, it is easy-to-read and fun to write.

# This first lab introduces the basics of Julia and some of its unique features in terms of creating custom types.
# This will become valuable in implementing automatic-differentiation for generic code, a feature that
# is particularly useful for SciML as it allows combining neural networks with general time-steppers.

# **Learning Outcomes**
#
# 1. Creating functions, both named and anonymous.
# 2. The notion of a type and how to make your own type.
# 3. Defining functions whose arguments are restricted to specific types.
# 4. Overloading functions like `+`, `*`, and `exp` for a custom type.
# 5. Construction of a dense `Vector` or `Matrix` either directly or via comprehensions or broadcasting.

# In what follows we need to use the testing package, which provides a macro called `@test`
# that error whenever a test returns false. We load thisas follows:

using Test

# ## 1.1 Functions in Julia

# We begin with creating functions, which can be done in a number of ways.
# The most standard way is using the keyword `function`, 
# followed by a name for the function, 
# and in parentheses a list of arguments.  
# Let's make a function that takes in a 
# single number $x$ and returns $x^2$.  

##

# There is also a convenient syntax for defining functions on
# one line, e.g., we can also write

##



# Multiple arguments to the function can be included with `,`.  
# Here's a function that takes in 3 arguments and returns the average.  
# (We write it on 3 lines only to show that functions can take multiple lines.)

##

# **Remark**: Julia is a compiled language: the first time we run a command on a specific type it compiles for that type.
# The second time it is called on a type it reuses the precompiled function.

#  **Warning**: Variables live in different scopes.  In the previous example, `x`, `y`, `z` and `ret` are _local variables_: 
# they only exist inside of `av`.  
# If you reference variables not defined inside the function, they will use the outer scope definition.  
# The following example shows that if we mistype the first argument as `xx`, 
# then it takes on the outer scope definition `x`, which is a complex number:


function av2(xx, y, z)
    (x + y + z)/3
end
x = 5
av2(2, 2, 2) # uses x = 5 from outside function

# You should avoid using global variables:
# we should ideally be able to predict the output of a function from knowing just the inputs.
# This also leads to faster code as the compiler can know the type of the variable.


# Here is an example of a more complicated function that computes
# the _(right-sided) rectangular rule_
# for approximating an integral: choose $n$ large so that
# $$
#   ∫_0^1 f(x) {\rm d}x ≈ {1 \over n} ∑_{j=1}^n f(j/n).
# $$
# It demonstrates that functions can take in  other functions as inputs
# and shows the syntax for a for-loop:

##


# ### Anonymous (lambda) functions

# It is possible to make unnamed anonymous functions,
# with two variants on syntax:

##

# There is not much difference between named and anonymous functions,
# both are compiled in the same manner. The only difference is
# named functions are in a sense "permanent". One can essentially think of
# named functions as "global constant anonymous functions".




# ----

# **Problem 1(a)** Complete the following function `leftrectangularrule(f, n)` That approximates
# an integral using the left-sided rectangular rule:
# $$
#   ∫_0^1 f(x) {\rm d}x ≈ {1 \over n} ∑_{j=0}^{n-1} f(j/n).
# $$

using Test # Loads `@test` again in case you didn't run the line above.

function leftrectangularrule(f, n)
    ## TODO: return (1/n) * ∑_{j=0}^{n-1} f(j/n) computed using a for-loop
    
end

@test leftrectangularrule(exp, 1000) ≈ exp(1) - 1 atol=1E-3 # tests that the approximation is accurate to 3 digits after the decimal point

# **Problem 1(b)** Use an anonymous function as input for `lefrectangularrule` to approximate
# the integral of $\cos x^2$ on $[0,1]$ to 5 digits.

## TODO: use an anonymous function to represent the function cos(x^2) and approximate its integral.

# ----

# ## 1.2 Types in Julia


# In compiled languages like Julia everything has a "type". The function `typeof` can be used to determine the type of,
# for example, a number.
# By default when we write an integer (e.g. `-123`) it is of type `Int`:

##

# On a 64-bit machine this will print `Int64`, where the `64` indicates it is using precisely 64 bits
# to represent the number. If we write something with
# a decimal point it represents a "real" number, whose storage is of type `Float64`:

##

# This is  a floating point number, and again the `64` indicates it is using precisely
# 64 bits to represent this number. Note that some operations involving `Int`s return `Float64`s:

##

# It is possible to have functions behave differently depending on the input type.
# To do so we can add a restriction denoted `::Int` or `::Float64` to the function "signature".
# Here we create a function `foo` that is equal to `1` if the input is an `Int`, `0` if the input is
# a `Float64`, and `-1` otherwise:

##


# The last line returns a list of `Int`s, which has the type `Tuple`.
# Note that there is a difference between an "integer" and the type `Int`: whilst 3.0 is an integer
# its type is `Float64` so `foo(3.0) == 0`.


# Types allow for combining multiple numbers (or instances of other types) to represent a more complicated
# object. A simple example of this is a complex number,
# which stores two real numbers $x$ and $y$ (either `Int` or `Float64` or indeed other real number types not yet discussed)
# to represent the complex number $x + {\rm i} y$. In Julia ${\rm i} = \sqrt{-1}$ is denoted `im` and
# hence we can create a complex number like $1+2{\rm i}$ as follows:

##

# This complex number has two "fields": the real and imaginary part. Accessing the fields is done
# using a `.`, here we display the real and imaginary parts as a `Tuple`:

##

# When we ask  its type we see it is a `Complex{Int}`:

##

# The `{Int}` indicates that each of the fields is an `Int`.
# Note we can add, subtract, multiply, or apply functions like `exp` to complex numbers:

##

# ### Supertypes

# Every type has a "supertype", which is an "abstract type": something you can't make an instance of it.
# For example, in the same way that "integers"
# are subsets of the "reals" we have that `Int` and `Float64` are subtypes of
# `Real`. Which is a subtype of `Number`. Which, as is everything, a subtype of `Any`. We can see this with the
# `supertype` function:

##

# -----
# **Problem 2(a)** Use `typeof` to determine the type of `1.2 + 2.3im`.

## TODO: What is the type of `1.2 + 2.3im`?



# **Problem 2(b)** Add another implementation of `foo` that returns `im` if the input
# is a `ComplexF64`.

## TODO: Overload foo for when the input is a ComplexF64 and return im


@test foo(1.1 + 2im) == im


# **Problem 3** Consider the Taylor series approximation to the exponential:
# $$
# \exp z ≈ ∑_{k=0}^n {z^k \over k!}
# $$
# Complete the function `exp_t(z, n)` that computes this and returns a
# `Complex{Float64}` if the input is complex and a `Float64` if the input is real.
# Do not use the inbuilt `factorial` function.
# Hint: It might help to think inductively: for $s_k = z^k/k!$ we have
# $$
#   s_{k+1}  = {z \over k+1} s_k.
# $$

function exp_t(z, n)
    ## TODO: Compute the first (n+1)-terms of the Taylor series of exp
    ## evaluated at z
    
end

@test exp_t(1.0, 10) isa Float64 # isa is used to test the type of a result
@test exp_t(im, 10) isa ComplexF64 # isa is used to test the type of a result

@test exp_t(1.0, 100) ≈ exp(1)


# ------

# ### Making our own Types


# One of the powerful parts of Julia is that it's very easy to make our own types. Lets begin with a simple
# implementation of a rational function $p/q$ where $p$ and $q$ are `Int`s.  Thus we want to create a new
# type called `Rat` with two fields `p` and `q` to represent the numerator and denominator, respectively.
# (For simplicity  we won't worry about restricting $p$ and $q$ to be `Int`.)
# We can construct such a type using the `struct` keyword:

##

# A new instance of `Rat` is created via e.g. `Rat(1, 2)` represents 1/2
# where the first argument specifies `p` and the second argument `q`.
# The fields are accessed by `.`:

##

# Unfortunately we can't actually do anything with this type, yet:

##

# The error is telling us to overload the `+` function when the inputs are both `Rat`.
# To do this we need to "import" the `+` function and then we can overload it like any
# other function:

##

# We can support mixing `Rat` and `Int` by adding additional functionality:

##

# -----

# **Problem 4** Support `*`, `-`, `/`, and `==` for `Rat` and `Int`.

## We import `+`, `-`, `*`, `/` so we can "overload" these operations
## specifically for `Rat`.
import Base: +, -, *, /, ==

## The ::Rat means the following version of `==` is only called if both
## arguments are Rat.
function ==(x::Rat, y::Rat)
    ## TODO: implement equality, making sure to check the case where
    ## the numerator/denominator are possibly reducible
    ## Hint: gcd and div may be useful. Use ? to find out what they do

    
end

## We can also support equality when `x isa Rat` and `y isa Int`
function ==(x::Rat, y::Int)
    ## TODO: implement
    
end

## TODO: implement ==(x::Int, y::Rat)


@test Rat(1, 2) == Rat(2, 4)
@test Rat(1, 2) ≠ Rat(1, 3)
@test Rat(2,2) == 1
@test 1 == Rat(2,2)

## TODO: implement +, -, *, and /,


@test Rat(1, 2) + Rat(1, 3) == Rat(5, 6)
@test Rat(1, 3) - Rat(1, 2) == Rat(-1, 6)
@test Rat(2, 3) * Rat(3, 4) == Rat(1, 2)
@test Rat(2, 3) / Rat(3, 4) == Rat(8, 9)


# ## 1.3 Arrays


# One can create arrays in multiple ways. For example, the function `zeros(Int, 10)` creates
# a 10-element `Vector` whose entries are all `zero(Int) == 0`. Or `fill(x, 10)` creates a
# 10-element `Vector` whose entries are all equal to `x`. Or you can use a comprehension:
# for example `[k^2 for k = 1:10]` creates a vector whose entries are `[1^2, 2^2, …, 10^2]`.
# This also works for matrices: `zeros(Int, 10, 5)` creates a 10 × 5 matrix of all zeros,
# and `[k^2 + j for k=1:3, j=1:4]` creates the following:

##

# Note sometimes it is best to create a vector/matrix and populate it. For example, the
# previous matrix could also been constructed as follows:

##

# **Remark** Julia uses 1-based indexing where the first index of a vector/matrix
# is 1. This is standard in all mathematical programming languages (Fortran, Maple, Matlab, Mathematica)
# whereas those designed for computer science use 0-based indexing (C, Python, Rust).



# Be careful: a `Matrix` or `Vector` can only ever contain entries of the right
# type. It will attempt to convert an assignment to the right type but will throw
# an error if not successful:

##


# ------
# **Problem 5(a)** Create a 5×6 matrix whose entries are `Int` which is
# one in all entries. Hint: use a for-loop, `ones`, `fill`, or a comprehension.
## TODO: Create a matrix of ones, 4 different ways


# **Problem 5(b)** Create a 1 × 5 `Matrix{Int}` with entries `A[k,j] = j`. Hint: use a for-loop or a comprehension.

## TODO: Create a 1 × 5  matrix whose entries equal the column, 2 different ways


# -------
# ### Transposes and adjoints

# We can also transpose a matrix `A` via `transpose(A)`
# or compute the adjoint (conjugate-transpose) via `A'` (which is
# equivalent to a transpose when the entries are real).
# This is done _lazily_: they return custom types `Transpose` or
# `Adjoint` that just wrap the input array and reinterpret the entries.
# Here is a simple example:

##

# If we change entries of `A'` it actually changes entries of `A` too since
# they are pointing to the same locations in memory, just interpreting the data differently:

##

# Note vector adjoints/transposes behave differently than 1 × n matrices: they are
# more like row-vectors. For example the following computes the dot product of two vectors:

##

# ### Broadcasting

# _Broadcasting_ is a powerful and convenient way to create matrices or vectors,
# where a function is applied to every entry of a vector or matrix.
# By adding `.` to the end of a function we "broadcast" the function over
# a vector:

##

# Broadcasting has some interesting behaviour for matrices.
# If one dimension of a matrix (or vector) is `1`, it automatically
# repeats the matrix (or vector) to match the size of another example.
# In the following we use broadcasting to pointwise-multiply a column and row
# vector to make a matrix:

##

# Since `size([1,2,3],2) == 1` it repeats the same vector to match the size
# `size([4,5]',2) == 2`. Similarly, `[4,5]'` is repeated 3 times. So the
# above is equivalent to:

A = [1 1;
     2 2;
     3 3] # same as [a a], i.e. repeat the vector a in each column
B = [4 5;
     4 5;
     4 5] # same as [b'; b' b'], i.e. repeat the row vector b' in each row

A .* B # equals the above a .* b'

# Note we can also use matrix broadcasting with our own functions:

##


# ### Ranges

# _Ranges_ are another useful example of vectors, but where the entries are defined "lazily" instead of
# actually created in memory.
# We have already seen that we can represent a range of integers via `a:b`. Note we can
# convert it to a `Vector` as follows:

Vector(2:6)

# We can also specify a step:

Vector(2:2:6), Vector(6:-1:2)

# Finally, the `range` function gives more functionality, for example, we can create 4 evenly
# spaced points between `-1` and `1`:

Vector(range(-1, 1; length=4))

# Note that `Vector` is mutable but a range is not:

r = 2:6
r[2] = 3   # Not allowed

# Both ranges `Vector` are subtypes of `AbstractVector`, whilst `Matrix` is a subtype of `AbstractMatrix`.


# -----

# **Problem 5(c)** Create a vector of length 5 whose entries are `Float64`
# approximations of `exp(-k)`. Hint: use a for-loop, broadcasting `f.(x)` notation, or a comprehension.
## TODO: Create a vector whose entries are exp(-k), 3 different ways


# ### Linear algebra

# Matrix-vector multiplication works as expected because `*` is overloaded:

A = [1 2;
     3 4;
     5 6]
x = [7, 8]
A * x

# We can also solve least squares problems using `\`:

b = randn(3)
A \ b # finds x that minimises norm(A*x - b)

# When a matrix is square, `\` reduces to a linear solve.

A = randn(5,5)
b = randn(5)
x = A \ b

@test A*x ≈ b
