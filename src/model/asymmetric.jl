struct Asymmetric <: Distribution
    d_l::Distribution
    d_r::Distribution
    n_l::Int
end

function _asymmetric_params(d::Asymmetric, θ)
    θ_l = θ[1:d.n_l] 
    θ_r = θ[d.n_l + 1:end] 
    return θ_l, θ_r
end

function _asymmetric_weights(d::Asymmetric, θ)
    θ_l, θ_r = _asymmetric_params(d, θ)

    # Assume that the densities are symmetric. Then the probabilities of the halves are 0.5.
    P_l = 0.5
    P_r = 0.5

    # Ensure continuity at origin by multiplying with the other density.
    w_l = logpdf(d.d_r, 0, θ_r)
    w_r = logpdf(d.d_l, 0, θ_l)
    
    # Normalise density.
    Z = exp(w_l) * P_l + exp(w_r) * P_r
    w_l = w_l - log(Z)
    w_r = w_r - log(Z)

    return (θ_l, w_l), (θ_r, w_r)
end

Asymmetric() = tup -> Asymmetric(tup, tup)

Asymmetric(tup::Tuple) = Asymmetric(tup, tup)

function Asymmetric(tup_l::Tuple, tup_r::Tuple)
    d_l, θ_l, ε_θ_l = tup_l
    d_r, θ_r, ε_θ_r = tup_r
    return (
        Asymmetric(d_l, d_r, length(θ_l)),
        vcat(θ_l, θ_r),
        vcat(ε_θ_l, ε_θ_r)
    )
end

function logpdf(d::Asymmetric, x, θ)
    (θ_l, w_l), (θ_r, w_r) = _asymmetric_weights(d, θ)
    return _if_then(
        x .< 0,
        w_l .+ logpdf(d.d_l, x, θ_l),
        w_r .+ logpdf(d.d_r, x, θ_r)
    )
end

function cdf(d::Asymmetric, x, θ)
    (θ_l, w_l), (θ_r, w_r) = _asymmetric_weights(d, θ)
    return _if_then(
        x .< 0,
        exp(w_l) .* cdf(d.d_l, x, θ_l),
        exp(w_l) * 0.5 .+ exp(w_r) .* (cdf(d.d_r, x, θ_r) .- 0.5)
    )
end

function rand(d::Asymmetric, n, θ)
    (θ_l, w_l), (θ_r, w_r) = _asymmetric_weights(d, θ)
    weights = ProbabilityWeights([exp(w_l), exp(w_r)])
    samples = Float64[]
    for i = 1:n
        if sample([true, false], weights)
            push!(samples, -abs(rand(d.d_l, 1, θ_l)[1]))
        else
            push!(samples, abs(rand(d.d_r, 1, θ_r)[1]))
        end
    end
    return samples
end

function display(d::Asymmetric, θ, ε, indent="")
    θ_l, θ_r = _asymmetric_params(d, θ)
    ε_l, ε_r = _asymmetric_params(d, ε)
    out = indent * "Asymmetric(\n"
    out *= display(d.d_l, θ_l, ε_l, indent * "    ")[1:end - 1] * ",\n"
    out *= display(d.d_r, θ_r, ε_r, indent * "    ")
    out *= indent * ")\n"
    return out
end
