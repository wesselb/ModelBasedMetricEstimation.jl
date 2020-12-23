using TailPickles
using Statistics
using Metrics
using PyPlot
using ProgressMeter
using LinearAlgebra
using ForwardDiff

import TailPickles: load_pjm_financials, Mixture, Gaussian, StudentsT, Location, Scale,
    Asymmetric
    
matplotlib.use("TkAgg")

data = load_pjm_financials()
t, gpf = data["GPF"]
_, ef = data["EF"]
x = gpf.return ./ gpf.volume

# Estimate location and width.
μ = median(x)
σ = std(x)

# Define the mixture model.
likelihood, θ₀, ε_θ₀ = Mixture(
        [(
            Gaussian()
                |> Scale(σ=σ / 2, ε_σ=σ / 2)
                |> Asymmetric()
                |> Location(μ=quantile(x, loc), ε_μ=σ / 2)
        ) for loc in [0.3, 0.7]]...,
        # Be vague about the guessed for the tail parameters. These could be way off!
        StudentsT(ν=1.5, ε_ν=10.0)
            |> Scale(σ=σ, ε_σ=10σ)
            |> Asymmetric()
            |> Location(μ=μ, ε_μ=10σ / sqrt(length(x))),
    ) 

# Use a log-Gaussian prior centred around the chosen parameters.
prior, α, _ = Gaussian() |> Scale{length(θ₀)}(σ=ε_θ₀ ./ 1.96) |> Location{length(θ₀)}(μ=θ₀)

# Perform MLE.
target(θ) = (
    sum(TailPickles.logpdf(likelihood, x, θ)) +
    sum(TailPickles.logpdf(prior, θ, α))
)
θ = TailPickles.lbfgs(target, θ₀, verbose=true)
ε_θ = 1.96sqrt.(diag(-inv(ForwardDiff.hessian(target, θ))))

println("Likelihood: ", target(θ), "\n")
println("θ₀:")
print(TailPickles.display(likelihood, θ₀, ε_θ₀))
println("\nθ:")
print(TailPickles.display(likelihood, θ, ε_θ))

# Make Q-Q plots.
ps = collect(1:length(x)) ./ (length(x) + 1)
model_quants = @showprogress "Computing model quantiles: " map(ps) do p
    TailPickles.icdf(likelihood, p, θ)
end;

function plot_inds(inds)
    err = 2sqrt.(ps .* (1 .- ps)) ./ TailPickles.pdf(likelihood, model_quants, θ) ./ sqrt(length(x))
    plot(sort(x)[inds], (sort(x) .- model_quants)[inds], ls="-", lw=0.5, marker="o", markersize=2)
    plot(sort(x)[inds], (sort(x) .- model_quants .- err)[inds], c="tab:red", ls="-", lw=0.5, marker="o", markersize=2)
    plot(sort(x)[inds], (sort(x) .- model_quants .+ err)[inds], c="tab:red", ls="-", lw=0.5, marker="o", markersize=2)
end

figure(figsize=(6, 4))

m = 20

subplot(2, 2, 1)
title("All")
plot_inds(1:length(x))
ylabel("Data Quantile Overshoot")
grid()

subplot(2, 2, 2)
title("Bulk")
plot_inds((m + 1):(length(x) - m))
grid()

subplot(2, 2, 3)
title("Left Tail")
plot_inds(1:m)
ylabel("Data Quantile Overshoot")
xlabel("Data Quantile")
grid()

subplot(2, 2, 4)
title("Right Tail")
plot_inds((length(x) - m + 1):length(x))
xlabel("Data Quantile")
grid()

tight_layout()

# Plot estimated likelihood.
figure(figsize=(6, 3))
x_plot = collect(range(minimum(x), maximum(x), length=1000))
hist(x, bins=200, density=true)
plot(x_plot, TailPickles.pdf(likelihood, x_plot, θ))
scatter(x, x .* 0, s=10, alpha=0.15, edgecolor="none")
title("Model Likelihood")
grid()
tight_layout()
show()

# Plot individual components.
π, θs = TailPickles._mixture_params(likelihood, θ)
figure(figsize=(6, 3))
x_plot = collect(range(-5, 5, length=1000))
hist(x, bins=200, density=true)
plot(x_plot, TailPickles.pdf(likelihood, x_plot, θ), c="tab:red", lw=2)
plot(x_plot, π[1] .* TailPickles.pdf(likelihood.ds[1], x_plot, θs[1]), c="tab:orange", lw=2, ls="--")
plot(x_plot, π[2] .* TailPickles.pdf(likelihood.ds[2], x_plot, θs[2]), c="tab:orange", lw=2, ls="--")
# plot(x_plot, π[3] .* TailPickles.pdf(likelihood.ds[3], x_plot, θs[3]), c="tab:orange", lw=2, ls="--")
plot(x_plot, π[3] .* TailPickles.pdf(likelihood.ds[3], x_plot, θs[3]), c="tab:orange", lw=2)
scatter(x, x .* 0, s=10, alpha=0.15, edgecolor="none")
xlim(-5, 5)
title("Model Likelihood")
grid()
xlabel("Return/volume")
ylabel("Density")
tight_layout()
savefig("components3.png", dpi=200)




