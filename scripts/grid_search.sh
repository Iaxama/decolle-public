#!/usr/bin/env bash

declare -a params=(mem_min script save_dir no_use_gpus help)

read_s_args()
{
    while (($#)) && [[ $1 != -* ]];  do sargs+=("$1"); shift; done
}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        if [[ ${params[*]} =~ $param ]]; then
          read_s_args "${@:2}"
          declare -a "$param"="(${sargs[*]})"
        fi
        if [[ ! ${params[*]} =~ $param ]]; then
            echo "Unknown argument $param "
            exit 1
        fi

   fi

  shift
done

mem_min=${mem_min-5000}

if [[ ! -v script ]]; then
  echo "Error: You must define a script to run. Run with --help for usage."
  exit 0
fi


if [[ ! -v save_dir ]]; then
  echo "Error: You must define a save directory to store results. Run with --help for usage."
  exit 0
fi

if [[ -v help ]]; then
    echo "Script to run several simulations with a set of parameters taken from different files."
    echo "The script verifies the presence of GPUs in the system and subsequently runs the required"
    echo "simulations in the one with most available memory. When no memory is available anymore it "
    echo "waits for previous simulations to stop before running the next ones."
    echo
    echo "Usage: grid_search.sh --script <TESTED SCRIPT> --save_dir <DIRECTORY TO SAVE RESULTS>"
    echo
    echo "Script parameters:"
    echo "    --script: script to test"
    echo "    --save_dir: subdirectory where logs and results are put"
    echo "    --mem_min: (optional) minimum GPU memory requirement per simulation (default to 5GB)"
    echo "    --no_use_gpus: (optional) list of GPU indices (as they appear in nvidia-smi) not to use"
    echo
    exit 0
fi

#Kill all child processes on exit or ctrl-c
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

#Check if and how many GPUs are available
if hash nvidia-smi 2>/dev/null; then
    NUM_GPUS=`nvidia-smi --query-gpu=count --format="csv,noheader" | head -n 1`
else
    NUM_GPUS=0
fi
echo "Found ${NUM_GPUS} GPUs."

if [ -v no_use_gpus ]; then
  for i in "${!no_use_gpus[@]}"; do
    if [[ "${no_use_gpus[i]}" -ge "$NUM_GPUS" ]]; then
      echo ${no_use_gpus[i]}
      unset 'no_use_gpus[i]'
    fi
  done
  if [ "${#no_use_gpus[@]}" == $NUM_GPUS ] ; then
    NUM_GPUS=0
  fi
fi

for filename in params_to_test/*; do
    if [ "${NUM_GPUS}" -gt "0" ] ;then
        
        # Look for available GPU
        while : ; do
            MAX_FREE_MEM=0
            
             for (( j=0; j<$NUM_GPUS; j++ )); do
                if [[ ${no_use_gpus[*]} =~ $j ]]; then
                   echo "Skipping GPU $j"
                   continue
                fi
                echo "Checking availability on GPU $j"
                MEM=0
                for i in {1..10}; do
                    ((MEM += `nvidia-smi --query-gpu=memory.free --format="csv,noheader,nounits" --id=${j}`))
                    sleep 1
                done
                ((MEM /= 10))
                if [ "$MEM" -gt "$MAX_FREE_MEM" ]; then
                    max_j=$j
                    MAX_FREE_MEM=$MEM
                fi
            done

            echo "The GPU with most free memory is GPU $max_j with $MAX_FREE_MEM MB"
            if [ "$MAX_FREE_MEM" -gt "$mem_min" ]; then # If more than 10GB are free the n use that GPU
                break
            else
                echo "You need more than $mem_min MB to launch the simulation."
            fi

            echo "All GPUS are busy. Waiting..."
            wait -n
        done

        # Running one simulation
        echo "Running on GPU number ${max_j}"
        [ ! -d "/path/to/dir" ] && mkdir -p logs/$save_dir
        CUDA_VISIBLE_DEVICES=${max_j} python -u $script --params_file ${filename} --save_dir $save_dir >& logs/$save_dir/`date +%d-%m-%Y_%H-%M-%S.log` &
	sleep 60

    else
        echo "Running on CPU"
        echo "python multilayer.py --params_file ${filename}"
    fi
done

echo "No more simulations to launch. Waiting for the ones currently running to finish"
wait # For the last simulations to finish
echo "All processes have finished their job. Exiting"

exit 0

