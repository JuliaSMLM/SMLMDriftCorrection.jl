using SMLMDriftCorrection
using SMLMSim
using SMLMData
using SMLMRender
using Documenter

DocMeta.setdocmeta!(SMLMDriftCorrection, :DocTestSetup, :(using SMLMDriftCorrection, SMLMData); recursive=true)

makedocs(;
    modules=[SMLMDriftCorrection],
    authors="klidke@unm.edu",
    repo=Documenter.Remotes.GitHub("JuliaSMLM", "SMLMDriftCorrection.jl"),
    sitename="SMLMDriftCorrection.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaSMLM.github.io/SMLMDriftCorrection.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Configuration" => "configuration.md",
        "Theory & Workflow" => "theory_workflow.md",
        "API" => "api.md",
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/JuliaSMLM/SMLMDriftCorrection.jl",
    devbranch="main",
)
