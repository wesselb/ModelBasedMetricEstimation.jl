import TailPickles: Gaussian, Laplace, StudentsT, Location, Scale, Asymmetric, ParetoTails,
    DifferentiableParetoTails, Mixture

@testset "Model"  begin
    for (sampleable, (d, θ, ε_θ)) in [
        (true,  Gaussian()),
        (true,  TailPickles.Laplace()),
        (true,  StudentsT(ν=2.0)),
        (true,  Gaussian() |> Scale(σ=2.0)),
        (true,  Gaussian() |> Scale(σ=2.0) |> Location(μ=0.2)),
        (true,  Gaussian() |> Scale(σ=2.0) |> Asymmetric()),
        (true,  Asymmetric(Gaussian() |> Scale(σ=2.0), StudentsT(ν=2.0))),
        (false, Gaussian() |> Scale(σ=2.0) |> ParetoTails(x₁=-1.5, α₁=1.0, x₂=1.5, α₂=1.0)),
        (false, Gaussian() |> Scale(σ=1.0) |> DifferentiableParetoTails(x=1.5)),
        (true,  Mixture(Gaussian() |> Scale(σ=2.0), StudentsT(ν=2.0)))
    ]
        println("Test case:")
        print(TailPickles.display(d, θ, ε_θ, " | "))
        
        # Check that the PDF integrates to one.
        numerical_intergral = solve(
            QuadratureProblem((x, _) -> TailPickles.pdf(d, x, θ), -Inf, Inf),
            QuadGKJL(), reltol=1e-8
        ).u
        @test numerical_intergral ≈ 1 rtol=1e-6
        
        for q in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
            # Check that the CDF agrees with the numerically integrated PDF.
            numerical_cdf = solve(
                QuadratureProblem((x, _) -> TailPickles.pdf(d, x, θ), -Inf, q),
                QuadGKJL(), reltol=1e-8
            ).u
            analytical_cdf = TailPickles.cdf(d, q, θ)
            @test numerical_cdf ≈ analytical_cdf rtol=1e-6

            # Check that inversion of CDF is correct.
            @test TailPickles.icdf(d, analytical_cdf, θ, x₀=0.9 * q, rtol=1e-8) ≈ q rtol=1e-4

            # Check that derivative of the inverse CDF is correct.
            numerical_grad = FiniteDifferences.grad(
                central_fdm(7, 1),
                θ′ -> TailPickles.icdf(d, analytical_cdf, θ′, x₀=0.9 * q, rtol=1e-8),
                θ
            )[1]
            analytical_grad = TailPickles.∇icdf(d, analytical_cdf, θ, x₀=0.9 * q, rtol=1e-8)
            @test numerical_grad ≈ analytical_grad rtol=1e-4
        end

        # Check that the CDF has the right limits.
        @test TailPickles.cdf(d, -1e3, θ) ≈ 0 atol=1e-3
        @test TailPickles.cdf(d, 1e3, θ) ≈ 1 rtol=1e-3

        if sampleable
            # Test ES and gradient.
            @test Metrics.es(rand(d, 1_000_000, θ)) ≈ TailPickles.es(d, 0.05, θ) rtol=5e-2
            @test Metrics.es(-rand(d, 1_000_000, θ)) ≈ TailPickles.es(d, 0.95, θ) rtol=5e-2
            for p in [0.05, 0.95]
                x₀ = quantile(rand(d, 1000, θ), p)
                numerical_grad = FiniteDifferences.grad(
                    central_fdm(7, 1), 
                    θ′ -> TailPickles.es(d, p, θ′, x₀=x₀, rtol=1e-8),
                    θ
                )[1]
                analytical_grad = TailPickles.∇es(d, p, θ, x₀=x₀, rtol=1e-8)
                @test numerical_grad ≈ analytical_grad rtol=1e-4
            end

            # Test expectation and gradient.
            f(x) = log(abs(x) + 1)
            mc_est = mean(f.(rand(d, 1_000_000, θ)))
            @test mc_est ≈ TailPickles.expectation(f, d, θ) rtol=1e-2
            numerical_grad = FiniteDifferences.grad(
                central_fdm(7, 1), 
                θ′ -> TailPickles.expectation(f, d, θ′, rtol=1e-8),
                θ
            )[1]
            analytical_grad = TailPickles.∇expectation(f, d, θ, rtol=1e-8)
            @test numerical_grad ≈ analytical_grad rtol=1e-4

            # Test two asymmetric expectations.
            g(x) = x > 0
            mc_est = mean(g.(rand(d, 1_000_000, θ)))
            @test mc_est ≈ TailPickles.expectation(g, d, θ) rtol=1e-2

            h(x) = x < 0
            mc_est = mean(h.(rand(d, 1_000_000, θ)))
            @test mc_est ≈ TailPickles.expectation(h, d, θ) rtol=1e-2
        end
    end
end

@testset "Student's T internals" begin
    @test TailPickles._studentst_ν(TailPickles._studentst_θ(1.5)) ≈ 1.5
end
