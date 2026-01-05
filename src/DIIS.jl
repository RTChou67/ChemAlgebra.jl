export DIISManager
export diis_update!
export gdiis_update!
mutable struct DIISManager{T}
	max_size::Int
	vecs::Vector{T}
	errs::Vector{T}
	function DIISManager{T}(max_size::Int = 10) where T
		new{T}(max_size, T[], T[])
	end
end
function _solve_coeffs(mgr::DIISManager)
	n = length(mgr.vecs)
	B = zeros(Float64, n, n)
	for i in 1:n
		for j in i:n
			val = _generic_dot(mgr.errs[i], mgr.errs[j])
			B[i, j] = B[j, i] = val
		end
	end
	size_A = n + 1
	A_mat = zeros(Float64, size_A, size_A)
	A_mat[1:n, 1:n] = B
	A_mat[1:n, n+1] .= -1.0
	A_mat[n+1, 1:n] .= -1.0
	rhs = zeros(Float64, size_A)
	rhs[n+1] = -1.0
	coeffs = pinv(A_mat) * rhs
	return coeffs[1:n]
end
function diis_update!(mgr::DIISManager, new_vec, new_err)
	_push_and_trim!(mgr, new_vec, new_err)
	if length(mgr.vecs) < 2
		return new_vec
	end
	c = _solve_coeffs(mgr)
	return _generic_combine(mgr.vecs, c)
end
function gdiis_update!(mgr::DIISManager, new_coords, new_grad)
	_push_and_trim!(mgr, new_coords, new_grad)
	if length(mgr.vecs) < 2
		return new_coords, new_grad
	end
	c = _solve_coeffs(mgr)
	q_star = _generic_combine(mgr.vecs, c)
	g_star = _generic_combine(mgr.errs, c)
	return q_star, g_star
end
function _push_and_trim!(mgr, v, e)
	push!(mgr.vecs, v)
	push!(mgr.errs, e)
	if length(mgr.vecs) > mgr.max_size
		popfirst!(mgr.vecs)
		popfirst!(mgr.errs)
	end
end
@inline _generic_dot(x::AbstractArray, y::AbstractArray) = dot(x, y)
function _generic_combine(vecs::Vector{T}, c::Vector{Float64}) where T <: AbstractArray
	res = c[1] .* vecs[1]
	for i in 2:length(c)
		res .+= c[i] .* vecs[i]
	end
	return res
end
function _generic_dot(x::Tuple, y::Tuple)
	s = 0.0
	for i in 1:length(x)
		s += dot(x[i], y[i])
	end
	return s
end
function _generic_combine(vecs::Vector{T}, c::Vector{Float64}) where T <: Tuple
	N_tuple = length(vecs[1])
	return ntuple(N_tuple) do k
		comp_vecs = [v[k] for v in vecs]
		_generic_combine(comp_vecs, c)
	end
end
