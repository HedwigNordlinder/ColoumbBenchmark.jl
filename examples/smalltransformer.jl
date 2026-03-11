using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/MurrellGroup/MurrellGroupRegistry"))
Pkg.add(url="https://github.com/HedwigNordlinder/Jester.jl")
Pkg.add(["Flowfusion", "ForwardBackward", "Flux", "RandomFeatureMaps", "Distributions", "StatsBase", "Plots", "CannotWaitForTheseOptimisers", "Onion"])
Pkg.instantiate()

using Flux, Flowfusion, ForwardBackward, RandomFeatureMaps, StatsBase, Plots
using CannotWaitForTheseOptimisers, Distributions, LinearAlgebra, Jester, Onion
using ColoumbBenchmark
using ProgressMeter
T = Float32
n_particles = 8
β = T(2.0)

sample_X1(n::Int) = reshape(T.(rand(CoulombGas(n_particles, β), n)), 1, n_particles, :)
sample_X0(n::Int) = sort(reshape(T.(rand(MvNormal(fill(-2.0, n_particles), I(n_particles)), n)), 1, n_particles, :), dims=2)

# --- Energy-based model ---

struct EnergyBasedModel{L}
    layers::L
end

Flux.@layer EnergyBasedModel

function EnergyBasedModel(embedding_dim::Int, n_transformers::Int)
    nheads = 4
    head_dim = 32
    layers = (;
        location_rff = RandomFourierFeatures(1 => 2embedding_dim, T(1.0)),
        location_rff_low_frequency = RandomFourierFeatures(1 => 2embedding_dim, T(0.1)),
        time_rff = RandomFourierFeatures(1 => 4embedding_dim, T(1.0)),
        time_embedding = Dense(4embedding_dim => embedding_dim, bias=false),
        location_encoder = Dense(4embedding_dim + 1 => embedding_dim, bias=false),
        transformers = [Onion.AdaTransformerBlock(embedding_dim, embedding_dim, nheads; head_dim=head_dim, qk_norm=true) for _ in 1:n_transformers],
        energy_decoder = Dense(embedding_dim => 1, bias=false),
    )
    return EnergyBasedModel(layers)
end

function (m::EnergyBasedModel)(t, x)
    # x: (1, n_particles, batch), t: (batch,)
    layers = m.layers
    loc_features = vcat(
        layers.location_rff(x),
        layers.location_rff_low_frequency(x),
        x,
    )
    h = layers.location_encoder(loc_features)

    t_cond = layers.time_embedding(layers.time_rff(reshape(t, 1, 1, :)))

    for layer in layers.transformers
        h = layer(h; cond=t_cond)
    end
    return layers.energy_decoder(h)  # (1, n_particles, batch)
end

# --- Training ---

P = BrownianMotion(T(0.15))
embedding_dim = 256
n_transformers = 2
model = EnergyBasedModel(embedding_dim, n_transformers)
opt_state = Flux.setup(Muon(), model)

n_samples = 1024
n_iters = 1000
losses = T[]

@showprogress for i in 1:n_iters
    X0 = ContinuousState(sample_X0(n_samples))
    X1 = ContinuousState(sample_X1(n_samples))
    t = rand(T, n_samples)
    Xt = bridge(P, X0, X1, t)

    loss, grad = Flux.withgradient(model) do m
        θ, re = Flux.destructure(m)
        energy_fn = (x, θ) -> sum(re(θ)(t, x))
        score = grad_fd(energy_fn, tensor(Xt), θ)
        t_scale = reshape(T(1.05) .- t, 1, 1, :)
        X̂₁ = ContinuousState(tensor(Xt) .+ score .* t_scale)
        floss(P, X̂₁, X1, scalefloss(P, t))
    end

    Flux.update!(opt_state, model, grad[1])
    push!(losses, loss)

    if i % 100 == 0
        println("Iter $i / $n_iters, loss: $(round(loss; digits=4))")
    end
end

# --- Generation ---

function gen_model(t, Xt)
    θ, re = Flux.destructure(model)
    x = tensor(Xt)
    n_batch = size(x, 3)
    t_vec = fill(T(t), n_batch)
    energy_fn = (x, θ) -> sum(re(θ)(t_vec, x))
    score = grad_fd(energy_fn, x, θ)
    t_scale = T(1.05) - T(t)
    ContinuousState(x .+ score .* t_scale)
end

n_gen = 500
X0_gen = ContinuousState(sample_X0(n_gen))
steps = T(0):T(0.005):T(1)
paths = Tracker()
samples = gen(P, X0_gen, gen_model, steps; tracker=paths)

# --- Visualization ---

# 1. Loss curve
p1 = plot(losses, xlabel="Iteration", ylabel="Loss", title="Training Loss", label=nothing, lw=1.5)
savefig(p1, joinpath(@__DIR__, "loss_curve.png"))

# 2. Particle trajectories
p2 = plot(title="Particle Trajectories", xlabel="Time", ylabel="Position")
n_show = min(20, n_gen)
for sample_idx in 1:n_show
    for particle_idx in 1:n_particles
        positions = [tensor(s[1])[1, particle_idx, sample_idx] for s in paths.xt]
        times = paths.t[1:length(positions)]
        plot!(p2, times, positions, alpha=0.3, label=nothing, color=particle_idx, lw=0.5)
    end
end
savefig(p2, joinpath(@__DIR__, "trajectories.png"))

# 3. Distribution comparison
true_samples = sample_X1(5000)
gen_positions = tensor(samples)

p3 = plot(title="Generated vs True Coulomb Gas", xlabel="Position", ylabel="Density")
histogram!(p3, vec(true_samples), normalize=:pdf, alpha=0.5, label="True CoulombGas", bins=80)
histogram!(p3, vec(gen_positions), normalize=:pdf, alpha=0.5, label="Generated", bins=80)
savefig(p3, joinpath(@__DIR__, "distribution_comparison.png"))

println("Done! Plots saved to examples/")
