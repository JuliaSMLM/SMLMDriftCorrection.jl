using SMLMDriftCorrection
using Documenter

DocMeta.setdocmeta!(SMLMDriftCorrection, :DocTestSetup, :(using SMLMDriftCorrection); recursive=true)

makedocs(;
    modules=[SMLMDriftCorrection],
    authors="klidke@unm.edu",
    repo="https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/blob/{commit}{path}#{line}",
    sitename="SMLMDriftCorrection.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaSMLM.github.io/SMLMDriftCorrection.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSMLM/SMLMDriftCorrection.jl",
    devbranch="main",
)
