struct Mixture <: Distribution
    ds::Vector{Distribution}
    ns::Vector{Int}
end

function _mixture_params(d::Mixture, θ)
    i = 1  # Track index in parameter vector.

    # Extract weights.
    π = exp.(θ[i:i + length(d.ds) - 1])
    i += length(d.ds)
    π = π ./ sum(π)

    # Extract parameters for components.
    θs = []
    for j = 1:length(d.ds)
        push!(θs, θ[i:i + d.ns[j] - 1])
        i += d.ns[j]
    end

    return π, θs
end

function _mixture_errors(d::Mixture, θ, ε)
    i = 1  # Track index in parameter vector.

    # Extract errors for weights.
    ε_π = ε[i:i + length(d.ds) - 1]
    i += length(d.ds)

    # Extract errors for components.
    εs = []
    for j = 1:length(d.ds)
        push!(εs, ε[i:i + d.ns[j] - 1])
        i += d.ns[j]
    end
    
    π, θs = _mixture_params(d, θ)

    return π, π .* (1 .- π) .* ε_π, θs, εs
end

Mixture(tup::Tuple) = tup  # There is no mixture in this case.
function Mixture(tups::Tuple...; π=ones(length(tups)), ε_π=0.2 .* ones(length(tups))) 
    π = π ./ sum(π)
    return (
        Mixture([tup[1] for tup in tups], [length(tup[2]) for tup in tups]),
        vcat(log.(π), [tup[2] for tup in tups]...),
        vcat(ε_π ./ π ./ (1 .- π), [tup[3] for tup in tups]...)
    )
end

function logpdf(d::Mixture, x, θ)
    π, θs = _mixture_params(d, θ)
    logpdfs = [log(πᵢ) .+ logpdf(dᵢ, x, θᵢ) for (πᵢ, dᵢ, θᵢ) in zip(π, d.ds, θs)]
    # TODO: Solve this in a better way! This only works for vectors and scalars.
    logpdfs = dropdims(_logsumexp(cat(logpdfs..., dims=2), dims=2), dims=2)
    ndims(x) == 0 && return logpdfs[1]
    return logpdfs
end

function cdf(d::Mixture, x, θ)
    π, θs = _mixture_params(d, θ)
    return sum([πᵢ .* cdf(dᵢ, x, θᵢ) for (πᵢ, dᵢ, θᵢ) in zip(π, d.ds, θs)])
end

function rand(d::Mixture, n, θ)
    π, θs = _mixture_params(d, θ)
    weights = ProbabilityWeights(π)
    samples = Float64[]
    for i = 1:n
        j = sample(1:length(d.ds), weights)
        push!(samples, rand(d.ds[j], 1, θs[j])[1])
    end
    return samples
end

function display(d::Mixture, θ, ε, indent="")
    π, ε_π, θs, εs  = _mixture_errors(d, θ, ε)
    out = indent * "Mixture(\n"
    for i in 1:length(θs)
        out *= display(d.ds[i], θs[i], εs[i], indent * "    ")[1:end - 1] * ",\n"
    end
    out *= indent * "    π=$(_round(π)),\n"
    out *= indent * "    ε_π=$(_round(ε_π))\n"
    out *= indent * ")\n"
    return out
end