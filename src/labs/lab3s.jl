# # SciML SANUM2024
# # Lab 3: Reverse-mode automatic differentiation and Zygote.jl

# When the number of unknowns becomes large forward-mode automatic differentiation as
# implemented in ForwardDiff.jl becomes prohibitively expensive and instead we need to
# use reverse-mode automatic differentiation: this is best thought of as implementing the chain-rule
# in an automatic fashion. 


# **Learning Outcomes**
#
# 1. Basics of reverse-mode automatic differentiation, "adjoint"s and back-propagation.
# 2. Implementation via Zygote.jl
# 3. Adding custom "adjoint"s. 



using Zygote

f = function(x)
    ret = zero(eltype(x))
    for k = 1:length(x)-1
        ret += x[k]*x[k+1]
    end
    ret
end

x = randn(5)
Zygote.gradient(f,x)

# The one catch is the complexity is quadratic:

@time Zygote.gradient(f,randn(1000));
@time Zygote.gradient(f,randn(10_000)); # around 100x slower