#!/bin/bash

# Define the base directory where the m12* directories are located
base_directory="/mnt/raid-project/murray/khullar/public_data_fire2"

# Get the list of directories to process, excluding m12i_res700
subdirectories=("m12c_res7100" "m12b_res7100" )
# Define the snapdir_* directories to process
snapdirs=("snapdir_312" "snapdir_356" "snapdir_382" "snapdir_412" "snapdir_534" "snapdir_591")

for subdir in "${subdirectories[@]}"; do
  directories="${base_directory}/${subdir}"

  # Iterate over each snapdir_* directory
  for snapdir in "${snapdirs[@]}"; do
    dir="${directories}/${snapdir}"
    
    if [[ -d "$dir" ]]; then
      echo "Processing snapdir: $dir"
      python3 CloudPhinder.py "$dir" --nmin=1 --alpha_crit=5 --outputfolder="${subdir}_output"
    else
      echo "Directory does not exist: $dir"
    fi
  done
done

echo "Processing completed"

