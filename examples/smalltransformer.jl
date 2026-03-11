using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/MurrellGroup/MurrellGroupRegistry"))
Pkg.add(url="https://github.com/HedwigNordlinder/Jester.jl")
Pkg.add(["Flowfusion", "ForwardBackward", "Flux", "RandomFeatureMaps", "Distributions", "StatsBase", "Plots", "CannotWaitForTheseOptimisers", "Onion"])
Pkg.instantiate()

using Flux, ForwardBackward, RandomFeatureMaps, StatsBase, Plots, CannotWaitForTheseOptimisers, Distributions, LinearAlgebra
using ColoumbBenchmark

T = Float32

sample_X1(n::Int; β = 1.0) = T.(rand(CoulombGas(2, β), n))
sample_X0(n::Int) = T.(rand(MvNormal([-2.0,-2.0], I(2)), n))

struct EnergyBasedModel{L}
    layers::L
end

Flux.@layer EnergyBasedModel
function EnergyBasedModel(embedding_dim::Int, n_transformers::Int)
    nheads = 8
    head_dim = 32
    layers = (;
             location_rff = RandomFourierFeatures(2 => 2embedding_dim),
             location_rff_low_frequency = RandomFourierFeatures(2 => 2embedding_dim), 
             time_rff = RandomFourierFeatures(1 => 4embedding_dim),
             time_embedding = Dense(4embedding_dim => embedding_dim, bias = false),
             location_encoder = Dense(4embedding_dim + 2 => embedding_dim, bias = false), # Add two since we concat actual location at end 
             transformers = [Onion.AdaTransformerBlock(embedding_dim, embedding_dim, nheads; head_dim = head_dim, qk_norm = true) for _ in 1:n_transformers], 
             energy_decoder = Dense(embedding_dim => 1, bias = false),)
    return EnergyBasedModel(layers)
end
function(m::EnergyBasedModel)(t,xState)
    Xt = xState.state
    layers = m.layers
    locations = tensor(Xt)
    x = layers.location_encoder(vcat(layers.location_rff(locations), layers.location_rff_low_frequency(locations), locations))
    t_cond = layers.time_embedding(layers.time_rff(t)) # This might be wierd/wrong, but I do not understand how the BranchingFlows toy example works
    for layer in layers.transformers
        x = layer(x; cond=t_cond)
    end
    return layers.energy_decoder(x)
end

