export bfgs_optimize
function bfgs_optimize(fg!, x_init::AbstractVector; tol = 1e-5, max_iter = 200, max_step = 0.2)
	n = length(x_init)
	x = copy(x_init)
	x_new = similar(x)
	g = similar(x)
	g_new = similar(x)
	p = similar(x)
	s = similar(x)
	y = similar(x)
	Hy = similar(x)
	H = Matrix{Float64}(I, n, n)
	E = fg!(g, x)
	history = Float64[]
	push!(history, E)
	for iter in 1:max_iter
		if norm(g) < tol
			return x, E, history
		end
		mul!(p, H, g)
		rmul!(p, -1.0)
		p_norm = norm(p)
		if p_norm > max_step
			scale_factor = max_step / p_norm
			rmul!(p, scale_factor)
		end
		alpha = 1.0
		c1 = 1e-4
		rho_ls = 0.5
		ls_success = false
		target_slope = c1 * dot(g, p)
		E_new = E
		for ls_iter in 1:20
			copyto!(x_new, x)
			axpy!(alpha, p, x_new)
			E_try = fg!(nothing, x_new)
			if isnan(E_try) || isinf(E_try)
				alpha *= 0.1
				continue
			end
			if E_try <= E + alpha * target_slope
				E_new = E_try
				fg!(g_new, x_new)
				ls_success = true
				break
			end
			alpha *= rho_ls
		end
		if !ls_success
			fill!(H, 0.0)
			for i in 1:n
				H[i, i] = 1.0
			end
			@. p = -g
			p_norm = norm(p)
			if p_norm > max_step
				rmul!(p, max_step / p_norm)
			end
			break
		end
		@. s = x_new - x
		@. y = g_new - g
		ys = dot(y, s)
		if ys > 1e-10
			rho = 1.0 / ys
			mul!(Hy, H, y)
			yHy = dot(y, Hy)
			term1 = rho + rho^2 * yHy
			BLAS.ger!(term1, s, s, H)
			BLAS.ger!(-rho, Hy, s, H)
			BLAS.ger!(-rho, s, Hy, H)
		end
		copyto!(x, x_new)
		copyto!(g, g_new)
		E = E_new
	end
	return x, E, history
end
