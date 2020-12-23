using TailPickles
using Statistics
using Metrics
using PyPlot
using QuadGK

import TailPickles: load_pjm_financials, StudentsT, Location, Scale, Asymmetric

matplotlib.use("TkAgg")

data = load_pjm_financials()
t, gpf = data["GPF"]
x = gpf.return ./ gpf.volume

# Define a model.
likelihood, θ₀ = Asymmetric(
    StudentsT(ν=1.5) |> Scale(σ=2.0),
    Gaussian() |> Scale(σ=1.0)
)  |> Location(μ=0.0)

# Contruct the target.
target(θ) = sum(TailPickles.logpdf(likelihood, x, θ))
θ = TailPickles.lbfgs(target, θ₀, verbose=true)

# Plot estimated likelihood.
figure()
x_plot = collect(range(minimum(x), maximum(x), length=500))
plot(x_plot, map(xᵢ -> TailPickles.pdf(likelihood, xᵢ, θ), x_plot))
scatter(x, x .* 0, s=10, alpha=0.15, edgecolor="none")
title("Estimate of likelihood")
show()