"""
# Main Command Line Script

This script is designed to be executed from the command line and serves as the entry point for processing auditory kernel data.
It takes several command-line arguments to configure the execution.

# Usage:
 - Example:  julia --check-bounds=yes main_command_line.jl "TIMIT" "tsv_files/train_TIMIT.tsv" 0.01 10000
"""

# Activate the current environment to ensure the correct dependencies are used
import Pkg
Pkg.activate("MyAuditoryKernels")

include(joinpath(@__DIR__, "filter_utils.jl"))  # Load the file containing the filter functions
include(joinpath(@__DIR__, "mp_utils.jl"))  # Load the file containing the MP functions

import DSP
import .mp_utils
import .filter_utils
using Random
using LinearAlgebra
using WAV
using CSV, DataFrames
using JLD2
using CUDA


function initialise_FIR(Filterparam, plot_flag=false)
    f = filter_utils.getFIRbandpassfilter(Filterparam.f_low, Filterparam.f_high, Filterparam.fs, Filterparam.length_filter)
    if plot_flag
        filter_utils.plotFIRresponse(f, Filterparam.fs, Filterparam.length_freq_ax)
    end
    return f
end


function initialise_kernels(MPparam)
    window = DSP.Windows.hamming(MPparam.kernel_size)
    initial_kernels = []
    for _ in 1:MPparam.Ng
        kernel = window .* CUDA.randn(Float64, MPparam.kernel_size) # Generate a random kernel of size (100,)
        kernel /= norm(kernel)                      # Normalize the kernel
        gradient = CUDA.zeros(MPparam.kernel_size)       # Initialize the gradient as zeros of size (100,)
        abs_amp = 0.0                               # Initialize absolute amplitude to 0.0

        push!(initial_kernels, mp_utils.Kernel(kernel, gradient, abs_amp))
    end
    return initial_kernels
end


function load_old_kernels(ID, count_start)
    dir_name = "Results/Results_" * ID
    file_name = "kernels_" * string(count_start) * ".jld2"
    file_path = joinpath(dir_name, file_name)
    try
        data = load(file_path, "kernels")
        kernels = [mp_utils.Kernel(CuArray(d.kernel), CuArray(d.gradient), d.abs_amp) for d in data]
        return kernels
    catch e
        println("Error loading kernels from file: ", file_path)
        println("Error details: ", e)
        return []  # Return an empty array if loading fails
    end
end


function load_audio(path)
    local x, fs_read, successLoadFlag
    try
        x, fs_read = wavread(path)
        successLoadFlag = true
    catch e
        x = nothing
        fs_read = nothing
        successLoadFlag = false
        println("Failed reading: ", path)
    end
    return x, fs_read, successLoadFlag
end


function filter_and_resample_audio(x, fs_read, Filterparam, filter_flag=false, normalise_flag=true)
    fs_read = Int(fs_read)
    if fs_read > Filterparam.fs
        println("resampling")
        x = DSP.Filters.resample(x, Filterparam.fs // fs_read, dims=1)
    end

    if filter_flag
        x = DSP.Filters.filt(f, x)
    end

    if normalise_flag
        x = x / maximum(abs.(x))
    end

    return x
end


function get_shuffled_paths(csv_file)
    df = CSV.read(csv_file, DataFrame)
    paths = shuffle(df.path_wav)
    return paths
end


function expand_kernels!(kernels, MPparam, count)
    if mod(count, MPparam.exp_update) == 0
        mp_utils.trim_and_expand_kernels!(kernels, MPparam.exp_threshold, MPparam.exp_range)
    end
end


function store_figure_and_jld2(kernels, ID, count, MPparam, Filterparam, csv_file, force_store=false)
    if mod(count, MPparam.nStore) == 0 || force_store
        # (1): store
        mp_utils.save_to_jld2(ID, count, MPparam, Filterparam, csv_file, kernels)

        # (2): plot
        mp_utils.arrayPlot(kernels, ID, count)
    end
end


function run_MP_epoch!(x, MPparam, kernels)
    x_res, kernel_list, amp_list, index_list, _ = mp_utils.matching_pursuit(x, MPparam.stop_type, MPparam.stop_cond, kernels, nothing, MPparam.max_iter)
    mp_utils.update_kernels!(index_list, kernel_list, amp_list, kernels, x_res, MPparam.step_size, MPparam.smoothing_weight)
end


function run_epochs(MPparam, Filterparam, kernels, csv_file, count_start, filter_flag, normalise_flag)
    println(" ")
    flush(stdout)
    count = 0

    # Outer loop: epochs - load csv with shuffled entries
    for _ in 1:MPparam.maxEpochs
        # Shuffle paths to audio
        shuffled_paths = get_shuffled_paths(csv_file)

        #Inner loop: loop over audio files.
        for path in shuffled_paths

            # Update step_size and exp_threshold based on schedules
            if count in MPparam.count_schedule
                idx = findfirst(==(count), MPparam.count_schedule)
                if idx !== nothing
                    MPparam.step_size = MPparam.step_size_schedule[idx]
                    MPparam.exp_threshold = MPparam.exp_threshold_schedule[idx]
                end
            end

            if count - 1 < count_start
                count += 1
            else
                println(" ")
                flush(stdout)

                println(count)
                println(path)

                x, fs_read, successLoadFlag = load_audio(path)

                if successLoadFlag
                    count += 1
                    x = filter_and_resample_audio(x, fs_read, Filterparam, filter_flag, normalise_flag)
                    run_MP_epoch!(x, MPparam, kernels)          # Run MP and gradient update
                    expand_kernels!(kernels, MPparam, count)    # Trim and expand the kernels every so often
                    store_figure_and_jld2(kernels, ID, count, MPparam, Filterparam, csv_file)  # Plot and store results every so often
                end
            end

            if count >= MPparam.nTrainIts
                store_figure_and_jld2(kernels, ID, count, MPparam, Filterparam, csv_file, true)  # Plot and store result
                @goto terminate_loops
            end
        end
    end
    @label terminate_loops
end


# Logging info
println("Number of threads: ", Threads.nthreads())
println(pwd())


##  Input arguments
ID = ARGS[1] #"TIMIT"
csv_file = ARGS[2] #"TIMIT_train.csv"
exp_threshold = parse(Float64, ARGS[3])
nTrainIts = parse(Int, ARGS[4])


## If there are 5 arguments we continue from a previous run
if length(ARGS) < 5
    continue_flag = false
    count_start = 0
else
    continue_flag = true
    count_start = parse(Int, ARGS[5])
end


##  Set user parameters
# matching pursuit
MPparam = mp_utils.MPparams(
    32,         # Ng (number of kernels)
    100,        # kernel_size
    10,         # random_seed
    "amplitude",# stop_type
    0.1,        # stop_cond
    40000,      # max_iter
    0.001,      # step_size
    0.0,        # smoothing_weight
    exp_threshold,      # exp_threshold (ARGS[3])
    1 / 10,       # exp_range
    50,          # exp_update
    25,          # nStore
    1000,        # maxEpochs (in practice we might hit 5 or something)
    [250, 500, 2500, 5000, 6000],     # count_schedule
    [0.0025, 0.005, 0.0025, 0.0025, 0.0025], # step_size_schedule
    exp_threshold * [1, 1, 0.75, 0.5, 0.25, 0.1],   # exp_threshold_schedule
    nTrainIts   # nTrainIts (ARGS[4])
)


# filter
Filterparam = filter_utils.Filterparams(
    100,        # f_low
    6000,       # f_high
    16000,      # fs
    256,        # length_filter
    1024,       # length_freq_ax (plotting only)
)


# Set random seed
Random.seed!(MPparam.random_seed)


# Initialise FIR filter (unused at the moment)
plot_flag = false       # Set to true if plotting is desired
filter_flag = false     # Set to true if filter is used
normalise_flag = true   # Set to true if normalisation is used
f = initialise_FIR(Filterparam, plot_flag)


# Initialise kernels
kernels = initialise_kernels(MPparam)


# If continue_flag: load old kernels (the reason we still initialised them is for the random seed)
if continue_flag
    kernels = load_old_kernels(ID, count_start)
end


# Main loop
mp_utils.arrayPlot(kernels, ID, count_start)
run_epochs(MPparam, Filterparam, kernels, csv_file, count_start, filter_flag, normalise_flag)


println("Maximum number of iterations reached. Terminating program.")
exit(0)
