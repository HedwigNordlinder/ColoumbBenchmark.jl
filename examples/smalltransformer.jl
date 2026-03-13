using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.Registry.add("General")
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/MurrellGroup/MurrellGroupRegistry"))
Pkg.add(url="https://github.com/HedwigNordlinder/Jester.jl")
Pkg.add(["Flowfusion", "ForwardBackward", "Flux", "RandomFeatureMaps", "Distributions", "StatsBase", "Plots", "Onion", "CUDA"])
Pkg.instantiate()

using Flux, Flowfusion, ForwardBackward, RandomFeatureMaps, StatsBase, Plots
using Distributions, LinearAlgebra, Jester, Onion
using ColoumbBenchmark
using ProgressMeter, CUDA, cuDNN
CUDA.device!(0)
T = Float32
n_particles = 100
βs = T[1.0, 2.0, 4.0]
locations = [-10, 0, 10]
coulomb_mixture = MixtureModel([CoulombGas(n_particles, βs[i], locations[i] .* ones(n_particles)) for i in eachindex(locations)])

sample_X1(n::Int) = reshape(T.(rand(coulomb_mixture, n)), 1, n_particles, :)
sample_X0(n::Int) = sort(reshape(T.(rand(MvNormal(fill(0, n_particles), I(n_particles)), n)), 1, n_particles, :), dims=2)

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
    h_pool = mean(h, dims=2)  
    return layers.energy_decoder(h_pool)  
end

# --- Training ---

P = BrownianMotion(T(0.15))
embedding_dim = 256
n_transformers = 5
model = EnergyBasedModel(embedding_dim, n_transformers) |> gpu

# --- Learning rate schedule: linear warmup + cosine decay ---
peak_lr = T(1e-3)
warmup_iters = 100
n_iters = 8000
n_samples = 1024

function lr_schedule(iter)
    if iter <= warmup_iters
        return peak_lr * iter / warmup_iters
    else
        progress = (iter - warmup_iters) / (n_iters - warmup_iters)
        return peak_lr * T(0.5) * (1 + cos(T(π) * progress))
    end
end

opt_state = Flux.setup(AdamW(peak_lr, (0.9, 0.999), 1e-2), model)
losses = T[]

@showprogress for i in 1:n_iters
    # Update learning rate
    Flux.adjust!(opt_state, T(lr_schedule(i)))

    X0 = ContinuousState(sample_X0(n_samples))
    X1 = ContinuousState(sample_X1(n_samples))
    t = rand(T, n_samples)
    Xt = bridge(P, X0, X1, t)

    Xt_gpu = gpu(tensor(Xt))
    X1_gpu = ContinuousState(gpu(tensor(X1)))
    t_gpu = gpu(t)

    loss, grad = Flux.withgradient(model) do m
        θ, re = Flux.destructure(m)
        energy_fn = (x, θ) -> sum(re(θ)(t_gpu, x))
        score = grad_fd(energy_fn, Xt_gpu, θ)
        t_scale = reshape(T(1.05) .- t_gpu, 1, 1, :)
        X̂₁ = ContinuousState(Xt_gpu .+ score .* t_scale)
        floss(P, X̂₁, X1_gpu, scalefloss(P, t_gpu))
    end

    Flux.update!(opt_state, model, grad[1])
    push!(losses, loss)

    if i % 100 == 0
        println("Iter $i / $n_iters, loss: $(round(loss; digits=4)), lr: $(round(lr_schedule(i); sigdigits=4))")
    end
end

# --- Generation ---

function gen_model(t, Xt)
    θ, re = Flux.destructure(model)
    x = gpu(tensor(Xt))
    n_batch = size(x, 3)
    t_vec = fill!(similar(x, n_batch), T(t))
    energy_fn = (x, θ) -> sum(re(θ)(t_vec, x))
    score = grad_fd(energy_fn, x, θ)
    t_scale = T(1.05) - T(t)
    ContinuousState(cpu(x .+ score .* t_scale))
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

# 2. Evolution GIF for a single sample
sample_idx = 1
snapshots = paths.xt
snap_times = paths.t[1:length(snapshots)]
xlims_range = let all_pos = vcat([cpu(tensor(s[1]))[1, :, sample_idx] for s in snapshots]...)
    (minimum(all_pos) - 1.0, maximum(all_pos) + 1.0)
end

anim = @animate for (k, s) in enumerate(snapshots)
    pos = cpu(tensor(s[1]))[1, :, sample_idx]
    scatter(pos, zeros(T, n_particles),
        xlims=xlims_range, ylims=(-0.5, 0.5),
        xlabel="Position", title="t = $(round(snap_times[k]; digits=3))",
        label=nothing, markersize=6, markerstrokewidth=0,
        yticks=[], size=(600, 200))
    vline!([locations...], linestyle=:dash, alpha=0.3, label=nothing, color=:gray)
end
gif(anim, joinpath(@__DIR__, "evolution.gif"), fps=30)

# 3. Distribution comparison: per-component marginals
true_samples = sample_X1(5000)
gen_positions = cpu(tensor(samples))

p3 = plot(title="Mixture of Shifted Coulomb Laws", xlabel="Position", ylabel="Density",
    size=(800, 400))
histogram!(p3, vec(true_samples), normalize=:pdf, alpha=0.4, label="True (mixture)", bins=100, color=:blue)
histogram!(p3, vec(gen_positions), normalize=:pdf, alpha=0.4, label="Generated", bins=100, color=:orange)
vline!(p3, [locations...], linestyle=:dash, alpha=0.5, label="Component centers", color=:black, lw=1.5)
savefig(p3, joinpath(@__DIR__, "distribution_comparison.png"))

# 4. Per-particle trajectory plot (overlay multiple samples)
n_show = min(50, n_gen)
p4 = plot(title="Particle Trajectories ($(n_show) samples)", xlabel="t", ylabel="Position",
    size=(800, 500), legend=:outertopright)
colors = palette(:tab10)
for si in 1:n_show
    for pi in 1:n_particles
        traj = [cpu(tensor(s[1]))[1, pi, si] for s in snapshots]
        plot!(p4, snap_times, traj, alpha=0.15, color=colors[(pi-1) % length(colors) + 1],
            label=(si == 1 ? "Particle $pi" : nothing), lw=0.5)
    end
end
hline!(p4, T.(locations), linestyle=:dash, alpha=0.5, label="Centers", color=:black, lw=1.5)
savefig(p4, joinpath(@__DIR__, "trajectories.png"))

println("Done! Plots saved to examples/")
