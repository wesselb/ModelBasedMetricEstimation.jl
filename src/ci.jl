"""
    ci(metric::Function, x)

Compute `metric` at `x` and an associated confidence interval.
"""
function ci end

function ci(::typeof(mean), x)
    μ = mean(x)
    ε = 1.96std(x) / sqrt(length(x))
    return μ, Interval{Closed, Closed}(μ - ε, μ + ε)
end

function ci(::typeof(median), x; α=0.05)
    d = Binomial(length(x), 0.5)
    _, index_first_zero = findmin(Distributions.pdf.(Ref(d), support(d)) .<= α / 2)
    index_last_one = index_first_zero - 1
    x_sorted = sort(x)
    return (
        median(x),
        Interval{Closed, Closed}(
            x_sorted[index_last_one],
            x_sorted[length(x) - index_last_one + 1]
        )
    )
end
