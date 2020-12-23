struct Linear{n} <: Distribution
    d::Distribution
end

_compress(x) = size(x) == (1,) ? x[1] : x

Linear(; β::Float64, ε_β::Float64=abs(β) / 2) = Linear{1}(β=β, ε_β=ε_β)
Linear{1}(; β::Float64, ε_β::Float64=abs(β) / 2) = 
    Initialiser(Linear{1}, θ -> vcat(β, θ), ε -> vcat(ε_β, ε))

logpdf(d::Linear{1}, x, θ; features) = 
    logpdf(d.d, _compress(x .- features .* θ[1]), θ[2:end])

cdf(d::Linear{1}, x, θ; features) = 
    cdf(d.d, _compress(x .- features .* θ[1]), θ[2:end])

rand(d::Linear{1}, n, θ; features) = features .* θ[1] .+ rand(d.d, n, θ[2:end])

function display(d::Linear{1}, θ, ε, indent="")
    β = θ[1]
    ε_β = ε[1]
    out = display(d.d, θ[2:end], ε[2:end], indent)[1:end - 1]
    out *= indent * " |> Linear(β=$(_round(β)), ε_β=$(_round(ε_β)))\n"
    return out
end

Linear{n}(; β::Vector{Float64}, ε_β::Vector{Float64}=abs.(β) ./ 2) where n = 
    Initialiser(Linear{n}, θ -> vcat(β, θ), ε -> vcat(ε_β, ε))

logpdf(d::Linear{n}, x, θ; features) where n = 
    logpdf(d.d, _compress(x .- features * θ[1:n]), θ[n + 1:end])

cdf(d::Linear{n}, x, θ; features) where n = 
    cdf(d.d, _compress(x .- features * θ[1:n]), θ[n + 1:end])
    
rand(d::Linear{n}, m, θ; features) where n = features * θ[1:n] .+ rand(d.d, m, θ[n + 1:end])

function display(d::Linear{n}, θ, ε, indent="") where n
    β = θ[1:n]
    ε_β = ε[1:n]
    out = display(d.d, θ[n + 1:end], ε[n + 1:end], indent)[1:end - 1]
    out *= " |> Linear{$n}(β=$(_round(β)), ε_β=$(_round(ε_β)))\n"
    return out
end