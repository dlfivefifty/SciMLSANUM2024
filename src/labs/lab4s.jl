# # SciML SANUM2024
# # Lab 4: Solving differential equations in DifferentialEquations.jl

# We now consider the solution of time-evolution ordinary differential equations (ODEs) using the
# DifferentialEquations.jl framework. An important feature is the ability to
# use automatic-differentiation with the numerical solutions, allowing us to solve
# simple nonlinear equations or optimisation problems involving parameters or initial conditions in the ODEs.

# **Learning Outcomes**
# 1. Solving ODEs using DifferentialEquations.jl
# 2. Differentiating an ODE with respect to parameters or initial conditions.
# 3. Solving simple nonlinear equations or optimisation problems.

using DifferentialEquations, Plots

# ## DifferentialEquations.jl

# Consider the pendulum equation with friction
# $$
# u'' = τ*u' - \sin u
# $$
# which we can rewrite as a first order system:
# $$
# [u᾽,v᾽] = [v, -τ*v - sin(u)]
# $$
# This is translated to an ODE as follows:


function pendulum(du, u, τ, t)
    du[1] = u[2]
    du[2] = -sin(u[1]) - τ*u[2]
end

τ = 0.0 # no friction
T = 10.0 # final time
u₀, v₀ = 1,1
prob = ODEProblem(pendulum, [u₀, v₀], (0.0, T), τ)
sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)

plot(sol)




# ## Combining auto-differentiation with DifferentialEquations.jl


using ForwardDiff
using ForwardDiff: derivative, gradient

function pendulumfriction(τ)
    T = 10.0 # final time
    u₀, v₀ = 1,1
    prob = ODEProblem(pendulum, [u₀, v₀], (0.0, T), τ)
    solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)
end

pendulumfrictionstop(τ) = pendulumfriction(τ)[end][1]


derivative(pendulumfrictionstop, 1.0)

# We can use this in a simple newton iteration to, for example, find the friction
# that results in a desired end conditon:


τ = 0.1
for k = 1:30
    τ = τ - derivative(pendulumfrictionstop, τ) \ pendulumfrictionstop(τ)
end
τ, pendulumfrictionstop(τ)

# We see that it has successed in finding one such friction so that we end 
# up at the bottom at the final time:

plot(pendulumfriction(τ))


using Zygote

function pendulum(du, u, τ, t)
    du[1] = u[2]
    du[2] = -sin(u[1]) - τ[1]*u[2]
end

function pendulumfriction(τ)
    T = 10.0 # final time
    u₀, v₀ = 1.0,1.0
    prob = ODEProblem(pendulum, [u₀, v₀], (0.0, T), τ)
    sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)
    sol[end][1]
end


Zygote.gradient(pendulumfriction, [1.0])

