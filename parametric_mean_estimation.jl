using TailPickles
using Statistics
using Metrics
using QuadGK

data = load_pjm_financials()
t, gpf = data["GPF"]
_, ef = data["EF"]
x = gpf.return ./ gpf.volume .- ef.return ./ ef.volume

# Define a model.
likelihood, θ₀, ε_θ₀ =
    StudentsT(ν=1.5) |>
    Scale(σ=1.0) |>
    Asymmetric() |>
    Location(μ=0.0, ε_μ=1.0)
prior, α = Gaussian() |> Scale{length(θ₀)}(σ=ε_θ₀) |> Location{length(θ₀)}(μ=θ₀)
print(TailPickles.display(likelihood, θ₀, ε_θ₀))

# Perform MAP and HMC to estimate the mean.
println("Empirical mean: ", ci(mean, x))
b_opt = Metrics.adaptive_block_size(mean, x, sizemin=10)
println("Sampled mean: ", mean(x), " ", Metrics.subsample_ci(mean, x, b_opt))
println(
    "Parametric Laplace estimate of mean: ",
    estimate_laplace(
        x,
        prior, α,
        likelihood, θ₀, ε_θ₀,
        expectation(identity, likelihood),
        verbose=false
    )
)
println(
    "Parametric Laplace estimate of prob.: ",
    estimate_laplace(
        x,
        prior, α,
        likelihood, θ₀, ε_θ₀,
        expectation(x -> x > 0, likelihood),
        verbose=false
    )
)
println(
    "Parametric HMC estimate of mean: ",
    estimate_hmc(
        x,
        prior, α,
        likelihood, θ₀, ε_θ₀,
        2000, 50,
        expectation(identity, likelihood),
        verbose=false
    )
)
println(
    "Parametric HMC estimate of prob.: ",
    estimate_hmc(
        x,
        prior, α,
        likelihood, θ₀, ε_θ₀,
        2000, 50,
        expectation(x -> x > 0, likelihood),
        verbose=false
    )
)
println(
    "Parametric HMC estimate of prob. of mean: ",
    estimate_hmc(
        x,
        prior, α,
        likelihood, θ₀, ε_θ₀,
        2000, 50,
        x -> expectation(identity, likelihood)(x) > 0,
        verbose=false
    )
)