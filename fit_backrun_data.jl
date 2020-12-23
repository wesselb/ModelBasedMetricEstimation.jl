using TailPickles
using Statistics
using Metrics
using ProgressMeter
using ForwardDiff
using LinearAlgebra

import TailPickles: Mixture, Scale, Location, Asymmetric, Gaussian, StudentsT

function fit_and_print(x)
    # Estimate location and width.
    μ = median(x)
    σ = median(abs.(x .- μ))

    # Define the mixture model.
    likelihood, θ₀, ε_θ₀ = Mixture(
        [(
            Gaussian()
                |> Scale(σ=σ / 2, ε_σ=σ / 2)
                |> Asymmetric()
                |> Location(μ=quantile(x, loc), ε_μ=σ)
        ) for loc in [0.5]]...,
        # Be vague about the guessed for the tail parameters. These could be way off!
        StudentsT(ν=2.0, ε_ν=6.0)
            |> Scale(σ=σ, ε_σ=σ)
            |> Asymmetric()
            |> Location(μ=μ, ε_μ=10σ / sqrt(length(x))),
        π=[1.0, 5.0]
    )

    # Use a log-Gaussian prior centred around the chosen parameters.
    prior, α, _ = Gaussian() |> Scale{length(θ₀)}(σ=ε_θ₀ ./ 1.96) |> Location{length(θ₀)}(μ=θ₀)

    # Perform MAP.
    target(θ) = (
        sum(TailPickles.logpdf(likelihood, x, θ)) +
        sum(TailPickles.logpdf(prior, θ, α))
    )
    θ = TailPickles.lbfgs(target, θ₀, verbose=false)
    ε_θ = 1.96sqrt.(diag(-inv(ForwardDiff.hessian(target, θ))))
    
    # Show results.
    print(TailPickles.display(likelihood, θ, ε_θ))
    TailPickles._show_plots(likelihood, θ, x)
end

df_pjm = TailPickles.load_pjm_ef_nl_financials()
df_miso = TailPickles.load_miso_ef_financials()

println("EF on MISO:")
fit_and_print(df_miso.return)

println("EF on PJM (NL):")
fit_and_print(df_pjm.return)
