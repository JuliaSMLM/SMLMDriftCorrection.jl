using Distributions


function gendata(;
    n_clusters=100,
    n_blink=5,
    photons=1000
)

    sz = 5
    σ_PSF = 1

    p_photons = Poisson(photons)

    n_c = rand(Poisson(n_clusters))
    n_b = zeros(Int, n_c)

    for i in 1:n_c
        n_b[i] = rand(Poisson(n_blink))
    end

    n = sum(n_b)
    x = zeros(Float32, n)
    y = zeros(Float32, n)
    σ_x = zeros(Float32, n)
    σ_y = zeros(Float32, n)

    cnt = 1
    for i in 1:n_c
        x0 = rand(Float32) * sz
        y0 = rand(Float32) * sz
        for j in 1:n_b[i]
            σ = σ_PSF / sqrt(min(rand(p_photons), 100))
            x[cnt] = x0 + σ * randn(Float32)
            y[cnt] = y0 + σ * randn(Float32)
            σ_x[cnt] = σ
            σ_y[cnt] = σ
            cnt += 1
        end
        
    end
    return x, y, σ_x, σ_y
end


