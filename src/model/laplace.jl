struct Laplace <: Distribution
    Laplace() = (new(), Float64[], Float64[])
end

logpdf(d::Laplace, x, θ) = -abs.(x) .- log(2)

cdf(d::Laplace, x, θ) = _if_then(x .> 0, 1 .- exp.(-abs.(x)) ./ 2, exp.(-abs.(x)) ./ 2)

display(::Laplace, θ, ε, indent="") = indent * "Laplace()\n"

rand(::Laplace, n, θ) = rand(Distributions.Laplace(), n)