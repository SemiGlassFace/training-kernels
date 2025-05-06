#!/bin/sh
#SBATCH --account=research-eemcs-insy
#SBATCH --partition=compute
#SBATCH --time=72:00:00       # Request run time (wall-clock). Default is 0 minute
#SBATCH --ntasks=1            # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=8     # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem-per-cpu=3G      # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --job-name=AudKer
#SBATCH --array=0-80:1 	
  
# Define array with language IDs
array_valid=(
	"TIMIT"
	"ab" "ar" 
	"ba" "bas" "be" "bg" "bn" "br" 
	"ca" "ckb" "cs" "cy" 
	"da" "dav" "de" "dv"
	"el" "en" "eo" "es" "et" "eu"
	"fa" "fi" "fr" "fy-NL"
	"gl"
	"ha" "hi" "hu" "hy-AM" 
	"ia" "id" "it"
	"ja"
	"ka" "kab" "kln" "kmr" "ky"
	"lg" "lt" "ltg" "luo" "lv"
	"mhr" "mk" "mn" "mr" "mrj" "mt"
	"nan-tw" "nl"
	"or"
	"pl" "ps" "pt"
	"rm-sursilv" "ro" "ru" "rw"
	"sah" "sk" "sq" "sv-SE" "sw"
	"ta" "th" "tok" "tr" "tt" 
	"ug" "uk" "ur" "uz"
	"vi"
	"yo" "yue" 
	"zh-CN" "zh-HK" "zh-TW"	
)

module load 2024r1 julia

# Set the number of threads for Julia
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run Julia script
echo "${array_valid[$SLURM_ARRAY_TASK_ID]}"
julia --check-bounds=yes main_command_line.jl "${array_valid[$SLURM_ARRAY_TASK_ID]}" "tsv_files/train_${array_valid[$SLURM_ARRAY_TASK_ID]}.tsv" 0.01 10000 > Logs/LOG_${array_valid[$SLURM_ARRAY_TASK_ID]}_$SLURM_ARRAY_TASK_ID.log

