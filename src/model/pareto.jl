struct UnnormalisedPareto <: Distribution end

_unnormalisedpareto_α(θ) = exp(θ[1])

logpdf(::UnnormalisedPareto, x, α) = -(α + 1) .* log.(abs.(x) .+ 1e-10)

logtailcdf(::UnnormalisedPareto, x, α) = -log(α) .- α .* log.(abs.(x) .+ 1e-10)

tailcdf(d::UnnormalisedPareto, x, α) = exp.(logtailcdf(d, x, α))

struct DifferentiableParetoTails <: Distribution
    d::Distribution
end

function _difftail_params(d::DifferentiableParetoTails, θ)
    x_m = exp(θ[1])
    θ_b = θ[2:end] # The cut-off point is parametrised.

    # Require differentiability: tail index.
    α = -1 - x_m * ForwardDiff.derivative(x -> logpdf(d.d, x[1], θ_b), x_m)
    α = max(α, 0.5)
   
    # Require continuity: weight of tail.
    w = logpdf(d.d, x_m, θ_b) - logpdf(UnnormalisedPareto(), x_m, α)

    # Normalise.
    Z = 2 * exp(w) * tailcdf(UnnormalisedPareto(), x_m, α) + 
        cdf(d.d, x_m, θ_b) - cdf(d.d, -x_m, θ_b)

    return x_m, α, w, θ_b, Z
end

DifferentiableParetoTails(; x::Float64, ε_x::Float64=1.0) = 
    Initialiser(
        DifferentiableParetoTails,
        θ -> vcat(log(x), θ),
        ε -> vcat([ε_x / x], ε)
    )

function logpdf(d::DifferentiableParetoTails, x, θ)
    x_m, α, w, θ_b, Z = _difftail_params(d, θ)
    return _if_then(
        abs.(x) .> x_m,
        w .+ logpdf(UnnormalisedPareto(), x, α),
        logpdf(d.d, x, θ_b)
    ) .- log(Z)
end

function cdf(d::DifferentiableParetoTails, x, θ)
    x_m, α, w, θ_b, Z = _difftail_params(d, θ)
    return _if_then2(
        x .< -x_m,
        x .< x_m,
        exp(w) .* tailcdf(UnnormalisedPareto(), x, α),
        exp(w) .* tailcdf(UnnormalisedPareto(), -x_m, α) .+ 
            cdf(d.d, x, θ_b) .- cdf(d.d, -x_m, θ_b),
        exp(w) * tailcdf(UnnormalisedPareto(), -x_m, α) .+
            cdf(d.d, x_m, θ_b) - cdf(d.d, -x_m, θ_b) .+
            exp(w) .* (
                tailcdf(UnnormalisedPareto(), x_m, α) .- 
                tailcdf(UnnormalisedPareto(), x, α)
            )
    ) ./ Z
end

function display(d::DifferentiableParetoTails, θ, ε, indent="")
    x_m, α, w, θ_b, Z = _difftail_params(d, θ)
    # TODO: Implement display of errors here.
    out = indent * "Tail: Pareto(x=$(x_m), α=$(α))\n"
    out *= display(d.d, θ_b, ε[2:end], indent)
    return out
end

struct ParetoTails <: Distribution
    d::Distribution
end

function _tail_params(θ)
    x₁ = -exp(θ[1])
    x₂ = exp(θ[2])
    α₁ = exp(θ[3])
    α₂ = exp(θ[4])
    return (x₁, α₁), (x₂, α₂), θ[5:end]
end

function _tail_weights(d::ParetoTails, θ)
    (x₁, α₁), (x₂, α₂), θ_b = _tail_params(θ)

    # Ensure continuity.
    w₁ = logpdf(d.d, x₁, θ_b) - logpdf(UnnormalisedPareto(), x₁, α₁)
    w₂ = logpdf(d.d, x₂, θ_b) - logpdf(UnnormalisedPareto(), x₂, α₂)

    # Normalise density.
    P₁ = tailcdf(UnnormalisedPareto(), x₁, α₁)
    P₂ = tailcdf(UnnormalisedPareto(), x₂, α₂)  
    Z = exp(w₁) * P₁ + exp(w₂) * P₂ + cdf(d.d, x₂, θ_b) - cdf(d.d, x₁, θ_b)

    return (w₁, P₁), (w₂, P₂), Z
end

ParetoTails(;
    x₁::Float64,
    ε_x₁::Float64=-x₁ / 2,
    x₂::Float64,
    ε_x₂::Float64=x₂ / 2,
    α₁::Float64,
    ε_α₁::Float64=α₁ / 2,
    α₂::Float64,
    ε_α₂::Float64=α₂ / 2
) = Initialiser(
    ParetoTails,
    θ -> vcat([log(-x₁), log(x₂), log(α₁), log(α₂)], θ),
    ε -> vcat([
        ε_x₁ / -x₁,
        ε_x₂ / x₂,
        ε_α₁ / α₁,
        ε_α₂ / α₂
    ], ε)
)

function logpdf(d::ParetoTails, x, θ)
    (x₁, α₁), (x₂, α₂), θ_b = _tail_params(θ)
    (w₁, P₁), (w₂, P₂), Z = _tail_weights(d, θ)
    return _if_then2(
        x .< x₁,
        x .< x₂,
        w₁ .+ logpdf(UnnormalisedPareto(), x, α₁),
        logpdf(d.d, x, θ_b),
        w₂ .+ logpdf(UnnormalisedPareto(), x, α₂)
    ) .- log(Z)
end

function cdf(d::ParetoTails, x, θ)
    (x₁, α₁), (x₂, α₂), θ_b = _tail_params(θ)
    (w₁, P₁), (w₂, P₂), Z = _tail_weights(d, θ)
    return _if_then2(
        x .< x₁,
        x .< x₂,
        exp(w₁) .* tailcdf(UnnormalisedPareto(), x, α₁),
        exp(w₁) * P₁ .+ cdf(d.d, x, θ_b) .- cdf(d.d, x₁, θ_b),
        exp(w₁) * P₁ .+ cdf(d.d, x₂, θ_b) - cdf(d.d, x₁, θ_b) .+
            exp(w₂) .* (P₂ .- tailcdf(UnnormalisedPareto(), x, α₂))
    ) ./ Z
end

function display(d::ParetoTails, θ, ε, indent="")
    (x₁, α₁), (x₂, α₂), θ_b = _tail_params(θ)
    # TODO: Implement display of errors here.
    out = indent * "Left tail: Pareto(x=$(_round(x₁)), α=$(_round(α₁)))\n"
    out *= indent * "Right tail: Pareto(x=$(_round(x₂)), α=$(_round(α₂)))\n"
    out *= display(d.d, θ_b, ε[5:end], indent)
    return out
end