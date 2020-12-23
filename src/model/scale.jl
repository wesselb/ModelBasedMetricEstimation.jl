struct Scale{n} <: Distribution
    d::Distribution
end

Scale(d::Distribution) = Scale{1}(d)

Scale(; σ::Float64, ε_σ::Float64=σ / 2) = Initialiser(
    Scale{1},
    θ -> vcat([log(σ)], θ),
    ε -> vcat([ε_σ / σ], ε)
)

logpdf(d::Scale{1}, x, θ) = logpdf(d.d, x ./ exp(θ[1]), θ[2:end]) .- θ[1]

cdf(d::Scale{1}, x, θ) = cdf(d.d, x ./ exp(θ[1]), θ[2:end])

rand(d::Scale{1}, n, θ) = exp(θ[1]) .* rand(d.d, n, θ[2:end]) 

function display(d::Scale{1}, θ, ε, indent="")
    σ = exp(θ[1])
    ε_σ = σ * ε[1]
    out = display(d.d, θ[2:end], ε[2:end], indent)[1:end - 1]
    out *= " |> Scale(σ=$(_round(σ)), ε_σ=$(_round(ε_σ)))\n"
    return out
end

Scale{n}(; σ::Vector{Float64}, ε_σ::Vector{Float64} = σ ./ 2) where n = 
    Initialiser(
        Scale{n},
        θ -> vcat(log.(σ), θ),
        ε -> vcat(ε_σ ./ σ, ε)
    )

logpdf(d::Scale{n}, x, θ) where n = 
    logpdf(d.d, x ./ exp.(θ[1:n]), θ[n + 1:end]) .- θ[1:n]
    
function display(d::Scale{n}, θ, ε, indent="") where n
    σ = exp.(θ[1:n])
    ε_σ = σ .* ε[1:n]
    out = display(d.d, θ[n + 1:end], ε[n + 1:end], indent)[1:end - 1]
    out *= " |> Scale{$n}(σ=$(_round(σ)), ε_σ=$(_round(ε_σ)))\n"
    return out
end