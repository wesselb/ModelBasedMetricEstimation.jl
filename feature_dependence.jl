using DataFrames
using Dates
using TailPickles
using Statistics
using Metrics
using ProgressMeter
using ForwardDiff
using LinearAlgebra
using PyPlot

import TailPickles: Mixture, Scale, Location, Asymmetric, Gaussian, StudentsT, Linear

df_pjm = TailPickles.load_pjm_ef_nl_financials()
df_temp = TailPickles.load_temp()["PJM"]
df_pjm_temp = innerjoin(df_pjm, select(df_temp, Not("time")), on="date")

df_pjm_temp = df_pjm_temp[df_pjm_temp.date .>= Date(2019, 1, 1), :]

# Prepare data.
x = df_pjm_temp.return ./ df_pjm_temp.volume

# Hand-engineered features:
features =  cat(
    # max.(quantile(df_pjm_temp.Tmin, 0.25) .- df_pjm_temp.Tmin, 0),
    max.(df_pjm_temp.Tmax .- quantile(df_pjm_temp.Tmax, 0.75), 0),
    dims=2
)
n = size(features, 2)
(n == 1) && (features = reshape(features, :))

# Guess coefficients with OLS.
β₀ = (features' * features) \ (features' * x)

# Estimate location and width.
μ = median(x)
σ = median(abs.(x .- μ))

# Define likelihood.
likelihood, θ₀, ε_θ₀ = Mixture(
    Gaussian()
        |> Scale(σ=σ / 2, ε_σ=σ / 2)
        |> Asymmetric()
        |> Location(μ=quantile(x, 0.5), ε_μ=σ / 2),
    # Be vague about the guessed for the tail parameters. These could be way off!
    StudentsT(ν=2.0, ε_ν=6.0)
        |> Scale(σ=σ, ε_σ=σ)
        |> Asymmetric()
        |> Location(μ=μ, ε_μ=10σ / sqrt(length(x))),
    π=[1.0, 3.0]
) |> Linear{n}(β=β₀, ε_β=2abs.(β₀))  # Flat prior on features.

# Use a log-Gaussian prior centred around the chosen parameters.
prior, α, _ = Gaussian() |> Scale{length(θ₀)}(σ=ε_θ₀ ./ 1.96) |> Location{length(θ₀)}(μ=θ₀)

# Perform MLE and show results.
target(θ) = (
    sum(TailPickles.logpdf(likelihood, x, θ, features=features)) +
    sum(TailPickles.logpdf(prior, θ, α))
)
θ = TailPickles.lbfgs(target, θ₀, verbose=true)
ε_θ = 1.96sqrt.(diag(-inv(ForwardDiff.hessian(target, θ))))
println("Likelihood: ", target(θ), "\n")
println("θ₀:")
print(TailPickles.display(likelihood, θ₀, ε_θ₀))
println("\nθ:")
print(TailPickles.display(likelihood, θ, ε_θ))
TailPickles._show_plots(likelihood, θ, x, features=features)

function _bin(x, y; chunks=100, chunk_size=nothing, chunk_reduction=mean, remove_zeros=false)
    inds = sortperm(x)
    x = x[inds]
    y = y[inds]
    
    x_binned = Float64[]
    y_binned = Float64[]
    
    isnothing(chunk_size) && (chunk_size = Int(round(length(x) / chunks)))
    i = 1
    while i < length(x)
        range = i:min(i + chunk_size - 1, length(x))
        if remove_zeros && chunk_reduction(x[range]) ≈ 0 
            i += chunk_size
            continue
        else
            push!(x_binned, chunk_reduction(x[range]))
            push!(y_binned, chunk_reduction(y[range]))
            i += chunk_size
        end
    end

    return x_binned, y_binned
end

if n > 1
    chunks = 40
    m = ceil(max(sqrt(n)))
    figure(figsize=(10, 8))
    for i = 1:(n + 1)
        subplot(m, m, i)
        if i == n + 1
            title("Dependence on All Features")
            scatter(_bin(features * θ[1:n], x, chunks=chunks, remove_zeros=true)..., s=10)
        else
            title("Dependence on Feature $i")
            scatter(_bin(features[:, i] .* θ[i], x, chunks=chunks, remove_zeros=true)..., s=10)
        end
        grid()
        i > m^2 - m && xlabel("Binned feature")
        mod(i, m) == 1 && ylabel("Binned profit")
    end
    tight_layout()
    show()
else
    figure(figsize=(6, 4))
    title("Dependence on Feature")
    scatter(_bin(features .* θ[1], x, chunk_size=10, remove_zeros=true)..., s=10)
    xlabel("Binned feature")
    ylabel("Binned profit")
    grid()
    tight_layout()
    show()
end