struct Gaussian <: Distribution
    Gaussian() = (new(), Float64[], Float64[])
end

logpdf(d::Gaussian, x, θ) = -(log(2π) .+ x.^2) ./ 2

cdf(d::Gaussian, x, θ) = (1 .+ erf.(x ./ sqrt(2))) ./ 2

display(::Gaussian, θ, ε, indent="") = indent * "Gaussian()\n"

rand(::Gaussian, n, θ) = randn(n)