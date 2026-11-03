# Guard entropy alignment against wandering on drift-corrected Genmab halves.
# Delivered July data live on /mnt/nas/cellpath/Genmab; skip without both NAS files.
using Test, SMLMDriftCorrection, JLD2

const DC = SMLMDriftCorrection
const ROOT = "/mnt/nas/cellpath/Genmab/Data/juliasmlm/2026-05-20_HeLaAlfaEGFR-DNAPAINT-tirf"

@testset "align_smld :entropy on real Genmab halves" begin
    fields = (
        ("HeLaAlfaEGFR_Resting", "HelaalfaEGFR_Resting_200kFrames1_2026_05_20_18_56_40"),
        ("HeLaAlfaEGFR_2F8-RGY+C1q", "HeLaAlfaEGFR_2F8_RGYC1q5min5ugml_2_2026_05_21_18_31_07"),
    )
    paths = [joinpath(ROOT, folder, field, "04_driftcorrect", "smld_corrected.jld2")
             for (folder, field) in fields]

    if !all(isfile, paths)
        @info "Skipping real Genmab half alignment: one or both NAS localization files are absent"
        @test_skip false
    else
        for ((_, field), path) in zip(fields, paths)
            @testset "$field" begin
                d = JLD2.load(path)
                smld = d[first(k for k in keys(d) if d[k] isa DC.SMLD)]
                midpoint = smld.n_frames ÷ 2
                first_half, second_half = map((true, false)) do early
                    emitters = deepcopy([e for e in smld.emitters
                                         if (e.frame <= midpoint) == early])
                    for e in emitters
                        e.dataset = 1
                    end
                    DC.BasicSMLD(emitters, smld.camera, smld.n_frames, 1,
                                 copy(smld.metadata))
                end
                (_, info) = align_smld([first_half, second_half]; method = :entropy)
                @test hypot(info.shifts[2]...) < 0.010
            end
        end
    end
end
