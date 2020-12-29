import MBME: Gaussian, Laplace, StudentsT, Location, Scale, Asymmetric, Mixture

@testset "Model"  begin
    for (sampleable, (d, θ, ε_θ)) in [
        (true,  Gaussian()),
        (true,  MBME.Laplace()),
        (true,  StudentsT(ν=2.0)),
        (true,  Gaussian() |> Scale(σ=2.0)),
        (true,  Gaussian() |> Scale(σ=2.0) |> Location(μ=0.2)),
        (true,  Gaussian() |> Scale(σ=2.0) |> Asymmetric()),
        (true,  Asymmetric(Gaussian() |> Scale(σ=2.0), StudentsT(ν=2.0))),
        (true,  Mixture(Gaussian() |> Scale(σ=2.0), StudentsT(ν=2.0)))
    ]
        println("Test case:")
        print(MBME.display(d, θ, ε_θ, " | "))

        # Check that the PDF integrates to one.
        numerical_intergral = solve(
            QuadratureProblem((x, _) -> MBME.pdf(d, x, θ), -Inf, Inf),
            QuadGKJL(), reltol=1e-8
        ).u
        @test numerical_intergral ≈ 1 rtol=1e-6

        for q in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
            # Check that the CDF agrees with the numerically integrated PDF.
            numerical_cdf = solve(
                QuadratureProblem((x, _) -> MBME.pdf(d, x, θ), -Inf, q),
                QuadGKJL(), reltol=1e-8
            ).u
            analytical_cdf = MBME.cdf(d, q, θ)
            @test numerical_cdf ≈ analytical_cdf rtol=1e-6

            # Check that inversion of CDF is correct.
            @test MBME.icdf(d, analytical_cdf, θ, x₀=0.9 * q, rtol=1e-8) ≈ q rtol=1e-4

            # Check that derivative of the inverse CDF is correct.
            numerical_grad = FiniteDifferences.grad(
                central_fdm(7, 1),
                θ′ -> MBME.icdf(d, analytical_cdf, θ′, x₀=0.9 * q, rtol=1e-8),
                θ
            )[1]
            analytical_grad = MBME.∇icdf(d, analytical_cdf, θ, x₀=0.9 * q, rtol=1e-8)
            @test numerical_grad ≈ analytical_grad rtol=1e-4
        end

        # Check that the CDF has the right limits.
        @test MBME.cdf(d, -1e3, θ) ≈ 0 atol=1e-3
        @test MBME.cdf(d, 1e3, θ) ≈ 1 rtol=1e-3

        if sampleable
            # Test ES and gradient.
            @test Metrics.es(rand(d, 1_000_000, θ)) ≈ MBME.es(d, 0.05, θ) rtol=5e-2
            @test Metrics.es(-rand(d, 1_000_000, θ)) ≈ MBME.es(d, 0.95, θ) rtol=5e-2
            for p in [0.05, 0.95]
                x₀ = quantile(rand(d, 1000, θ), p)
                numerical_grad = FiniteDifferences.grad(
                    central_fdm(7, 1),
                    θ′ -> MBME.es(d, p, θ′, x₀=x₀, rtol=1e-8),
                    θ
                )[1]
                analytical_grad = MBME.∇es(d, p, θ, x₀=x₀, rtol=1e-8)
                @test numerical_grad ≈ analytical_grad rtol=1e-4
            end

            # Test expectation and gradient.
            f(x) = log(abs(x) + 1)
            mc_est = mean(f.(rand(d, 1_000_000, θ)))
            @test mc_est ≈ MBME.expectation(f, d, θ) rtol=1e-2
            numerical_grad = FiniteDifferences.grad(
                central_fdm(7, 1),
                θ′ -> MBME.expectation(f, d, θ′, rtol=1e-8),
                θ
            )[1]
            analytical_grad = MBME.∇expectation(f, d, θ, rtol=1e-8)
            @test numerical_grad ≈ analytical_grad rtol=1e-4

            # Test two asymmetric expectations.
            g(x) = x > 0
            mc_est = mean(g.(rand(d, 1_000_000, θ)))
            @test mc_est ≈ MBME.expectation(g, d, θ) rtol=1e-2

            h(x) = x < 0
            mc_est = mean(h.(rand(d, 1_000_000, θ)))
            @test mc_est ≈ MBME.expectation(h, d, θ) rtol=1e-2
        end
    end
end

@testset "Student's T internals" begin
    @test MBME._studentst_ν(MBME._studentst_θ(1.5)) ≈ 1.5
end
