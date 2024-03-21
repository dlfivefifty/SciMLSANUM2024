# # SciML SANUM2024
# # Lab 4: Solving differential equations in DifferentialEquations.jl


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
    sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)
    sol[end]
end


derivative(pendulumfriction, 1.0)

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