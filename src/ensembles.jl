struct GaussianPotential end
(::GaussianPotential)(x) = x^2 / 2

struct CoulombGas{T<:Real,F,M<:AbstractVector} <: ContinuousMultivariateDistribution
    n::Int
    β::T
    V::F
    μ::M
end

CoulombGas(n::Int, β::Real) = CoulombGas(n, β, GaussianPotential())
CoulombGas(n::Int, β::Real, V) = CoulombGas(n, β, V, zeros(typeof(β), n))
CoulombGas(n::Int, β::Real, μ::AbstractVector) = CoulombGas(n, β, GaussianPotential(), μ)

Base.length(d::CoulombGas) = d.n

function Distributions._logpdf(d::CoulombGas, x::AbstractVector)
    n = length(x)
    # Interaction term: β ∑_{i<j} log|xᵢ - xⱼ|
    interaction = zero(eltype(x))
    @inbounds for i in 1:n, j in (i+1):n
        interaction += log(abs(x[i] - x[j]))
    end
    # Confining potential: -∑ᵢ V(xᵢ)
    confinement = sum(xi -> d.V(xi), x .- d.μ)
    return d.β * interaction - confinement
end

# Exact sampling via Dumitriu-Edelman tridiagonal model (Gaussian β-ensemble)
function Distributions._rand!(rng::AbstractRNG, d::CoulombGas{<:Real,GaussianPotential}, x::AbstractVector)
    n = d.n
    β = d.β
    # Tridiagonal matrix B with diagonal ~ N(0,2) and off-diagonal ~ χ_{β(n-k)}
    a = [randn(rng) * sqrt(2) for _ in 1:n]
    b = [sqrt(rand(rng, Chisq(β * (n - k)))) for k in 1:n-1]
    # Eigenvalues of B have density ∝ ∏|λᵢ-λⱼ|^β exp(-∑λ²/4)
    # Rescale by 1/√2 to match V(x) = x²/2 convention
    λ = eigvals(SymTridiagonal(a, b))
    x .= λ ./ sqrt(2) .+ d.μ
    return x
end
