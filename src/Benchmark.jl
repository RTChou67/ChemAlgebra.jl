# src/Benchmark.jl

function benchmark()
	Random.seed!(1234)
	n_roots = 4

	function measure(f)
		try
			return (@belapsed $f() samples=100) * 1000
		catch
			return NaN
		end
	end

	println("\n" * "="^90)
	@printf("%-10s | %-6s | %-10s | %-10s | %-10s | %-10s | %s\n",
		"Type", "N", "ChemAlg", "Arpack", "Krylov", "LOBPCG", "Speedup/Winner")
	println("-"^90)

	scenarios = [
		("Near-Diag", [1000, 5000, 10000, 20000], N -> begin
			D = spdiagm(0 => sort(rand(N)) .* 50.0)
			R = sprand(N, N, max(0.001, 10.0/N))
			D + 0.01 * (R + R')
		end),
		("Random", [500, 1000, 2000], N -> begin
			A = randn(N, N)
			(A + A') / 2
		end),
	]

	for (label, dims, mat_gen) in scenarios
		for N in dims
			H = mat_gen(N)

			X0 = zeros(N, n_roots)
			for i in 1:n_roots
				X0[i, i] = 1.0
			end
			x0 = X0[:, 1]
			P  = issparse(H) ? Diagonal(1.0 ./ diag(H)) : I

			if N == dims[1]
				# 预热不使用分号，展开写
				try
					Davidson(H, n_roots, max_iter = 2)
				catch
				end
			end

			t_my = measure(() -> Davidson(H, n_roots, tol = 1e-6))

			t_ar = measure(() -> eigs(H, nev = n_roots, which = :SR, v0 = x0, tol = 1e-6))

			t_kr = measure(() -> KrylovKit.eigsolve(H, x0, n_roots, :SR, tol = 1e-6))

			t_lo = measure(() -> IterativeSolvers.lobpcg(H, false, X0, P = P, tol = 1e-6))

			times = [t_my, t_ar, t_kr, t_lo]
			best = minimum(filter(!isnan, times))

			win_mark = ""
			if best == t_my
				win_mark = "ChemAlg 🚀"
			elseif best == t_lo
				win_mark = "LOBPCG"
			else
				win_mark = "Other"
			end

			fmt(t) = isnan(t) ? "FAIL" : @sprintf("%.2f", t)

			@printf("%-10s | %-6d | %10s | %10s | %10s | %10s | %s\n",
				label, N, fmt(t_my), fmt(t_ar), fmt(t_kr), fmt(t_lo), win_mark)
		end
		println("-"^90)
	end
end

function benchmark_diis()
	Random.seed!(1234)
	dims = [100, 300, 500]
	hist = 8

	scf_step(F) = 0.95 .* F .+ 0.05 .* sin.(F)
	nlsolve_f!(s, x) = (s .= x .- scf_step(x))

	println("\n" * "="^60)
	@printf("%-6s | %-10s | %-10s | %s\n", "N", "ChemAlg", "NLsolve", "Speedup")
	println("-"^60)

	for N in dims
		F0 = rand(N, N)

		function run_chem(F_in)
			mgr = DIISManager{Matrix{Float64}}(hist)
			F = copy(F_in)
			for _ in 1:100
				err = scf_step(F) - F
				if norm(err) < 1e-6
					break
				end
				F = diis_update!(mgr, F, err)
			end
			F
		end

		t_chem = (@belapsed $run_chem($F0) samples=100) * 1000

		t_nl = (@belapsed nlsolve($nlsolve_f!, $F0, method = :anderson, m = $hist, ftol = 1e-6) samples=100) * 1000

		@printf("%-6d | %10.2f | %10.2f | x%.1f 🚀\n",
			N, t_chem, t_nl, t_nl / t_chem)
	end
	println("="^60)
end
