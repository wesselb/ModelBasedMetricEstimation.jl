abstract type Distribution end

struct Initialiser{T}
    t::T
    init_θ::Function
    init_ε::Function
end

(init::Initialiser)(d::Distribution) =
    init.t(d), init.init_θ(Float64[]), init.init_ε(Float64[])
(init::Initialiser)(x::Tuple{Distribution, Vector{Float64}, Vector{Float64}}) =
    (init.t(x[1]), init.init_θ(x[2]), init.init_ε(x[3]))
    
_round(x::Real) = @sprintf("%.3e", x)
function _round(x::Vector)
    out = "[$(_round(x[1]))"
    for xᵢ in x[2:end]
        out *= ", $(_round(xᵢ))"
    end
    out *= "]"
    return out
end

# Fallbacks:
logpdf(d, x, θ; features::Nothing) = logpdf(d, x, θ)
pdf(d::Distribution, x, θ; features=nothing) = exp.(logpdf(d, x, θ, features=features))
cdf(d, x, θ; features::Nothing) = cdf(d, x, θ)
rand(d, n, θ; features::Nothing) = rand(d, n, θ)

icdf(d::Distribution, p, θ; x₀=0.0, features=nothing, rtol=1e-8) =
    _invert(
        x -> cdf(d, x, θ, features=features),
        x -> pdf(d, x, θ, features=features),
        x₀,
        p,
        rtol=rtol
    )
function ∇icdf(d::Distribution, p, θ; x₀=0.0, trunc=1e10, features=nothing, rtol=1e-8)
    x = icdf(d, p, θ, x₀=x₀, features=features, rtol=rtol)
    if abs(x) >= trunc
        @warn("∇icdf got large number from inversion. Returning zero gradient.")
        return zeros(length(θ))
    end
    return (
        -ForwardDiff.gradient(θ′ -> cdf(d, x, θ′, features=features), θ)
        ./ pdf(d, x, θ, features=features)
    )
end

function es(d::Distribution, p, θ; rtol=1e-6, x₀=0.0, trunc=1e10, features=nothing)
    q = icdf(d, p, θ, x₀=x₀, features=features, rtol=rtol / 100)
    if q >= trunc
        @warn("es got large quantile. Returning scaled expectation.")
        return (
            expectation(identity, d, θ, rtol=rtol, trunc=trunc, features=features)
            / (p < 0.5 ? p : 1 - p)
        )
    end
    σ = std(rand(d, 100, θ, features=features)) / 10
    return (p < 0.5 ? -1 : 1) * solve(
        QuadratureProblem(
            (x, _) -> (abs(x) >= trunc ? 0 : σ^2 * x * pdf(d, σ * x, θ, features=features)),
            p < 0.5 ? -Inf : q / σ,
            p < 0.5 ? q / σ : Inf
        ),
        QuadGKJL(), reltol=rtol
    ).u / (p < 0.5 ? p : 1 - p)
end
function ∇es(d::Distribution, p, θ; rtol=1e-6, x₀=0.0, trunc=1e10, features=nothing)
    q = icdf(d, p, θ, x₀=x₀, features=features, rtol=rtol / 100)
    if q >= trunc
        @warn("∇es got large quantile. Returning scaled ∇expectation.")
        return (
            ∇expectation(identity, d, θ, rtol=rtol, trunc=trunc, features=features)
            / (p < 0.5 ? p : 1 - p)
        )
    end
    ∇q = ∇icdf(d, p, θ, x₀=x₀, features=features, rtol=rtol / 100)
    σ = std(rand(d, 100, θ, features=features)) / 10
    integrand(x) = 
        σ^2 .* x .* ForwardDiff.gradient(θ′ -> pdf(d, σ * x, θ′, features=features), θ)
    ∇integral = solve(
        QuadratureProblem(
            (x, _) -> (abs(x) >= trunc ? zeros(length(θ)) : integrand(x)),
            p < 0.5 ? -Inf : q / σ,
            p < 0.5 ? q / σ : Inf
        ),
        QuadGKJL(), reltol=rtol
    ).u
    return (p < 0.5 ? -1 : 1) .* (
        ∇integral .+ (p < 0.5 ? 1 : -1) .* q .* pdf(d, q, θ, features=features) .* ∇q
    ) ./ (p < 0.5 ? p : 1 - p)
end

function expectation(f, d::Distribution, θ; rtol=1e-8, trunc=1e10, features=nothing)
    σ = std(rand(d, 100, θ, features=features)) / 10
    integrand(x, θ) = σ * f(σ * x) * pdf(d, σ * x, θ, features=features)
    return solve(QuadratureProblem(
        (x, θ′) -> abs(x) >= trunc ? 0 : integrand(x, θ′), -Inf, Inf, θ
    ), QuadGKJL(), reltol=rtol).u
end
function ∇expectation(f, d::Distribution, θ; rtol=1e-8, trunc=1e10, features=nothing)
    σ = std(rand(d, 100, θ, features=features)) / 10
    integrand(x, θ) = 
        σ * f(σ * x) * ForwardDiff.gradient(θ′ -> pdf(d, σ * x, θ′, features=features), θ)
    return solve(QuadratureProblem(
        (x, θ′) -> abs(x) >= trunc ? zeros(length(θ)) : integrand(x, θ′), -Inf, Inf, θ
    ), QuadGKJL(), reltol=rtol).u
end
