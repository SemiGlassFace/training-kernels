module mp_utils

using LinearAlgebra
using DSP
using Base.Threads
using FileIO: load, save, loadstreaming, savestreaming
using JLD2
# using Plots
using CSV, DataFrames
using CUDA

export Kernel, MPparams             # structures
export matching_pursuit, reconstruct_matching_pursuit# matching pursuit
export update_kernels!, trim_and_expand_kernels! # kernel learning
export save_to_jld2                # loggin and storitng


# Define a struct to hold kernel data
mutable struct Kernel
    kernel::CuArray{Float64}         # The actual kernel (1D array)
    gradient::CuArray{Float64}       # The gradient of the kernel (1D array)
    abs_amp::Float64                 # The absolute amplitude (float)
end


# Define a struct to hold parameters related to matching pursuit and gradient updates
mutable struct MPparams
    Ng::Int32               # The number of kernels
    kernel_size::Int32      # The initial kernel length
    random_seed::Int32      # The absolute amplitude (float)
    stop_type::String       # The type for which matching pursuit runs )"amplitude" vs "iterations"
    stop_cond::Float64      # The point at which to stop the stop type (it is int for iterations, so not sure how that works)
    max_iter::Int32         # The maximum allowed number of iterations
    step_size::Float64      # The update step size
    smoothing_weight::Float64 # The weight with which the gradients is smoothed
    exp_threshold::Float64  # The norm for which to expand the kernels
    exp_range::Float64      # The range on which the norm should be computed
    exp_update::Int32       # After how many iterations the expansion is done
    nStore::Int32           # How often to store the kernels
    maxEpochs::Int32        # Maximum number of epochs
    count_schedule::Vector{Int32}  # Schedule for updating step_size and exp_threshold
    step_size_schedule::Vector{Float64}  # Schedule for updating step_size
    exp_threshold_schedule::Vector{Float64}  # Schedule for updating exp_threshold
    nTrainIts::Int32        # Number of training iterations
end


function matching_pursuit(x, stop_type, stop_cond, kernels, x_res=nothing, max_iter=nothing)
    if isnothing(x_res)
        x_res = copy(x)
    end
    if isnothing(max_iter)
        max_iter = -1
    end

    stop_type = string(stop_type)

    # Initial conditions
    amp_list = Float64[]
    index_list = Int[]
    kernel_list = Any[]
    nIt = 0
    norm_list = [norm(x_res)]

    if stop_type == "amplitude"
        # MP iterations
        while true
            # amp_val, index_val, kernel_val, x_res = matching_pursuit_iter_jit(x_res, typed_kernels)
            amp_val, index_val, kernel_val, x_res = matching_pursuit_iter(x_res, kernels)
            push!(amp_list, amp_val)
            push!(index_list, index_val)
            push!(kernel_list, kernel_val)
            push!(norm_list, norm(x_res))
            #println(amp_list[end])
            if abs(amp_list[end]) < stop_cond
                break
            end
            if max_iter > 0
                if nIt > max_iter
                    println("Maximum number of MP iterations reached")
                    break
                end
            end
            nIt += 1
        end

    elseif stop_type == "iterations"
        # MP iterations
        while nIt < stop_cond
            amp_val, index_val, kernel_val, x_res = matching_pursuit_iter(x_res, kernels)
            push!(amp_list, amp_val)
            push!(index_list, index_val)
            push!(kernel_list, kernel_val)
            push!(norm_list, norm(x_res))
            nIt += 1
        end
    else
        println("Invalid stop condition specified, choose 'iterations' or 'amplitude'")
    end

    return x_res, kernel_list, amp_list, index_list, norm_list
end


@inline function process_kernel(kernel, x_res_flip)
    tmp = similar(x_res_flip, length(x_res_flip) + length(kernel) - 1)
    conv!(tmp, x_res_flip, kernel)
    b_j = argmax(abs.(tmp))
    b_j = b_j[1]
    a_j = tmp[b_j]
    return a_j, b_j
end


function matching_pursuit_iter(x_res, kernels)
    Ng = length(kernels)  # number of kernels
    a = CUDA.zeros(Ng)
    b = CUDA.zeros(Int, Ng)

    x_res_flip = reverse!(x_res)

    # Preallocate results array
    results = CuArray{Tuple{Float64,Int}}(undef, Ng)  # Assuming YourType is the type of a_j
    @threads for j in 1:Ng
        # Call the process_kernel function and store the result
        results[j] = process_kernel(kernels[j].kernel, x_res_flip)
    end

    # Unpack the results into a and b
    for (j, (a_j, b_j)) in enumerate(results)
        a[j] = a_j
        b[j] = b_j
    end

    # Find the kernel with the highest absolute value in a
    kernel_val = argmax(abs.(a))
    index_val = length(x_res) - b[kernel_val] + 1
    amp_val = a[kernel_val]

    # Get the kernel to subtract
    kernel_tmp = amp_val * kernels[kernel_val].kernel

    # Try to subtract the kernel from the signal
    try
        x_res[index_val:index_val+length(kernel_tmp)-1] .-= kernel_tmp
    catch
        # Handle case where kernel is larger than the remaining signal length
        if index_val > 0
            x_res[index_val:end] .-= kernel_tmp[1:length(x_res[index_val:end])]
        elseif index_val < 1
            index_end = index_val + length(kernels[kernel_val].kernel)
            x_res[1:index_end] .-= kernel_tmp[end-length(x_res[1:index_end])+1:end]
        end
    end

    return amp_val, index_val, kernel_val, x_res
end


function reconstruct_matching_pursuit(index_list, kernel_list, amp_list, kernels, x=nothing)
    # Get maximum kernel length
    Nk_list = [length(k.kernel) for k in kernels]  # Preallocate Nk_list
    Nk_max = maximum(Nk_list)

    # Determine the size of x_hat
    if x === nothing
        x_hat = CUDA.zeros(maximum(index_list) + Nk_max)
    else
        x_hat = CUDA.zeros(size(x))
    end

    # Reconstruct the signal
    for i in eachindex(amp_list)
        Nk = length(kernels[kernel_list[i]].kernel)
        Indx1 = index_list[i]
        Indx2 = Indx1 + Nk - 1

        try
            x_hat[Indx1:Indx2] .+= amp_list[i] * kernels[kernel_list[i]].kernel
        catch e
            if Indx1 <= 0
                println("Index out of bounds: Indx1 = $Indx1")  # Handle exception
            else
                println("Unexpected indexing error at Indx1 = $Indx1")
            end
        end
    end

    return x_hat
end


function update_kernels!(index_list, kernel_list, amp_list, kernels, x_res, step_size, smoothing_weight1)
    # Sort the lists based on kernel number
    sorted_indices = sortperm(kernel_list)
    kernel_list = kernel_list[sorted_indices]
    index_list = index_list[sorted_indices]
    amp_list = amp_list[sorted_indices]

    # Iterate over the kernels
    Ng = length(kernels)
    for ng in 1:Ng
        # Find all indices corresponding to certain type of kernel
        INDX = findall(kernel_list .== ng)

        # If there are any uses for this kernel:
        if !isempty(INDX)
            lKernel = length(kernels[ng].kernel)
            grad = zeros(lKernel)
            amp_norm = 0.0
            for indx in INDX
                indx1 = index_list[indx]
                indx2 = index_list[indx] + lKernel - 1

                if indx1 > 0 && indx2 <= length(x_res)
                    grad_step = amp_list[indx] * x_res[indx1:indx2]
                    grad .+= grad_step
                    amp_norm += abs(amp_list[indx])
                else
                    # TODO, include exceptions --> Ignored. The effect of the boundaries on the kernels is likely very limited.
                end
            end

            # Keep track of total amplitude in kernel updates
            kernels[ng].abs_amp += amp_norm

            # Note: We use the biased momentum. With smoothing_weight = 0.7, 10 iterations are needed to get to 0.7^10=0.03 (note that for expanded edges the counter restarts)
            kernels[ng].gradient = (1 - smoothing_weight1) * grad + smoothing_weight1 * kernels[ng].gradient
            kernels[ng].kernel += step_size * kernels[ng].gradient
            kernels[ng].kernel /= norm(kernels[ng].kernel)
        end
    end
end


function trim_and_expand_kernels!(kernels, threshold, expansion_range)
    # kernels is a vector of structs containing the kernels
    # threshold is the threshold on which expansion/reduction is based
    # expansion range is the length of the subset of the kernel on which we decide whether or not to expand.

    Ng = length(kernels)

    # expanding kernels
    for ng in 1:Ng
        Lg = length(kernels[ng].kernel)   # length kernel
        nPad = min(floor(Int, 0.1 * Lg), 20)  # padding length

        # Update length
        kernels[ng].kernel = vcat(zeros(nPad), kernels[ng].kernel, zeros(nPad))
        kernels[ng].gradient = vcat(zeros(nPad), kernels[ng].gradient, zeros(nPad))
    end

    # Trimming kernels
    for ng in 1:Ng
        kernel = kernels[ng].kernel
        gradient = kernels[ng].gradient
        exp_indx = ceil(Int, length(kernels[ng].kernel) * expansion_range)
        while (norm(kernel[1:exp_indx])^2 < threshold) || (norm(kernel[end-exp_indx+1:end])^2 < threshold)
            if (norm(kernel[1:exp_indx])^2 < threshold)
                kernel = kernel[2:end]
                gradient = gradient[2:end]
            end
            if (norm(kernel[end-exp_indx+1:end])^2 < threshold)
                kernel = kernel[1:end-1]
                gradient = gradient[1:end-1]
            end
        end

        kernel = kernel / norm(kernel)

        # update
        kernels[ng].kernel = kernel
        kernels[ng].gradient = gradient
    end
end

# ##  Function for plotting
# function arrayPlot(kernels, ID::String, count::Int)
#     Ng = length(kernels)  # Number of kernels
#     rows, cols = 4, 8     # Define layout size
#     max_plots = rows * cols
#     Ng = min(Ng, max_plots)  # Prevent exceeding 32 subplots

#     # Construct the file path where the figure will be saved
#     dir_name = "Results/Results_" * ID
#     file_name = "figure_" * string(count) * ".svg"
#     file_path = joinpath(dir_name, file_name)

#     if !isdir(dir_name)
#         mkdir(dir_name)
#     end

#     # Create plot with a more tightly packed layout
#     p = plot(
#         layout=(rows, cols),  # 4x8 grid layout
#         size=(1200, 600),      # Figure size (adjust as needed)
#         margin=0.5Plots.mm,    # Tight margin to reduce whitespace
#         padding=0.5Plots.mm,   # Reducing padding between subplots
#         legend=false,          # Disable legend for clarity
#         showaxis=false,        # Hide axes to save space
#         framestyle=:none,      # No borders for each plot
#     )

#     # Add each kernel as a subplot (adjusting subplot numbers to fit)
#     for j in 1:Ng
#         plot!(p, Array(kernels[j].kernel), subplot=j, showaxis=false, legend=false)
#     end

#     # Save the plot to the specified file as SVG
#     savefig(p, file_path)

#     println("Plot saved to: ", file_path)  # Print where the file is saved
#     p = nothing
# end


## Function for saving result
function save_to_jld2(ID::String, count::Int, MPparam, Filterparam, csv_file::String, kernels)
    # Create directory if it doesn't exist
    dir_name = "Results/Results_" * ID
    if !isdir(dir_name)
        mkdir(dir_name)
    end

    # Construct file path
    file_name = "kernels_" * string(count) * ".jld2"
    file_path = joinpath(dir_name, file_name)

    # Save variables to JLD2 file
    JLD2.@save file_path MPparam Filterparam csv_file count kernels

    println("Saved to: ", file_path)
end
end
