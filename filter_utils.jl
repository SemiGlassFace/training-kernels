module filter_utils

using DSP
using CUDA

export Filterparams, getFIRbandpassfilter, plotFIRresponse

mutable struct Filterparams
    f_low::Float64
    f_high::Float64
    fs::Int32
    length_filter::Int32
    length_freq_ax::Int32
end


# https://weavejl.mpastell.com/v0.2/examples/FIR_design.html
function FIRfreqz(b::CuArray, w=range(0, stop=π, length=1024))
    n = length(w)
    h = CuArray{ComplexF32}(undef, n)
    sw = 0
    for i = 1:n
        for j = 1:length(b)
            sw += b[j] * exp(-im * w[i])^-j
        end
        h[i] = sw
        sw = 0
    end
    return h
end


function getFIRbandpassfilter(f_low, f_high, fs, length_filter)
    responsetype = Bandpass(f_low, f_high)
    designmethod = FIRWindow(hamming(length_filter))
    f = digitalfilter(responsetype, designmethod; fs=fs)
    return f
end


function plotFIRresponse(f, fs, length_freq_ax)
    w = collect(range(0, pi, length=length_freq_ax))
    H = FIRfreqz(f, w)
    HdB = 20 * log10.(abs.(H))
    ws = w / pi * fs / 2
    display(plot(ws, HdB))  #xlabel = "Frequency (Hz)", ylabel = "Magnitude (db)"))

    H_phase = unwrap(-atan.(imag(H), real(H)))
    display(plot(ws, H_phase)) # label = "Frequency (Hz)" ,ylabel = "Phase (radians)"))
end
end
