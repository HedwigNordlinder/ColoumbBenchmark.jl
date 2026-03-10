using ColoumbBenchmark
using Documenter

DocMeta.setdocmeta!(ColoumbBenchmark, :DocTestSetup, :(using ColoumbBenchmark); recursive=true)

makedocs(;
    modules=[ColoumbBenchmark],
    authors="HedwigNordlinder <hedwignordlinder@gmail.com> and contributors",
    sitename="ColoumbBenchmark.jl",
    format=Documenter.HTML(;
        canonical="https://HedwigNordlinder.github.io/ColoumbBenchmark.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HedwigNordlinder/ColoumbBenchmark.jl",
    devbranch="main",
)
