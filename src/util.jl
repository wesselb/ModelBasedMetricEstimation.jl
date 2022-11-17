function _logsumexp(x; dims=:)
    u = ignore_derivatives(maximum(x, dims=dims))
    return log.(sum(exp.(x .- u), dims=dims)) .+ u
end

function _if_then(c, a, b)
    c = ignore_derivatives(c)
    return c .* a .+ (1 .- c) .* b
end

function _if_then2(c₁, c₂, a, b, c)
    c₁ = ignore_derivatives(c₁)
    c₂ = ignore_derivatives(c₂)
    return c₁ .* a .+ (1 .- c₁) .* c₂ .* b .+ (1 .- c₁) .* (1 .- c₂) .* c
end

function _invert(f, ∇f, x, y; rtol=1e-6, atol=1e-12, trunc=1e12)
    f_x = f(x)

    max_its = 200
    it = 0

    while true
        # Determine Newton step.
        ∇f_x = ∇f(x)
        step = -(f_x - y) / ∇f_x

        # Take step.
        x += step
        f_x = f(x)
        
        # Record errors.
        ε_x = 2abs(step)
        ε_y = abs(f_x - y)

        # Check for convergence.
        (ε_y < rtol * abs(y) || ε_y < atol) && (ε_x < rtol * abs(x) || ε_x < atol) && break

        # Truncate infinities.
        if abs(x) >= trunc
            @warn("Inversion yielded large number. Truncating.")
            return trunc * sign(x)
        end
        
        # Do not take too many steps.
        it += 1
        if it == max_its
            @warn(
                "Failed to converge in $(max_its) iterations. " *
                "Returning with errors $(ε_x) and $(ε_y). " *
                "Current gradient is $(∇f_x)."
            )
            return x
        end
    end
    return x
end
