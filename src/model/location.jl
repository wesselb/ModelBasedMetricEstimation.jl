struct Location{n} <: Distribution
    d::Distribution
end

Location(d::Distribution) = Location{1}(d)

Location(; μ::Float64, ε_μ::Float64=abs(μ) / 2) =
    Initialiser(Location{1}, θ -> vcat([μ], θ), ε -> vcat([ε_μ], ε))

logpdf(d::Location{1}, x, θ) = logpdf(d.d, x .- θ[1], θ[2:end])

cdf(d::Location{1}, x, θ) = cdf(d.d, x .- θ[1], θ[2:end])

rand(d::Location{1}, n, θ) = θ[1] .+ rand(d.d, n, θ[2:end])

function display(d::Location{1}, θ, ε, indent="")
    μ = θ[1]
    ε_μ = ε[1]
    out = display(d.d, θ[2:end], ε[2:end], indent)[1:end - 1]
    out *= " |> Location(μ=$(_round(μ)), ε_μ=$(_round(ε_μ)))\n"
    return out
end

Location{n}(; μ::Vector{Float64}, ε_μ::Vector{Float64}=abs.(μ) ./ 2) where n = 
    Initialiser(Location{n}, θ -> vcat(μ, θ), ε -> vcat(ε_μ, ε))

logpdf(d::Location{n}, x, θ) where n = logpdf(d.d, x .- θ[1:n], θ[n + 1:end])

function display(d::Location{n}, θ, ε, indent="") where n
    μ = θ[1:n]
    ε_μ = ε[1:n]
    out = display(d.d, θ[n + 1:end], ε[n + 1:end], indent)[1:end - 1]
    out *= " |> Location{$n}(μ=$(_round(μ)), ε_μ=$(_round(ε_μ)))\n"
    return out
end