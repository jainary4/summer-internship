#!/bin/bash 
directories=(
  "/mnt/raid-project/murray/khullar/public_data_fire2/m12i_res7100/snapdir_312"
  "/mnt/raid-project/murray/khullar/public_data_fire2/m12i_res7100/snapdir_356"
  "/mnt/raid-project/murray/khullar/public_data_fire2/m12i_res7100/snapdir_382"
  "/mnt/raid-project/murray/khullar/public_data_fire2/m12i_res7100/snapdir_412"
  "/mnt/raid-project/murray/khullar/public_data_fire2/m12i_res7100/snapdir_534"
  "/mnt/raid-project/murray/khullar/public_data_fire2/m12i_res7100/snapdir_591"
)

for dir in "${directories[@]}"; do
  echo "Processing directory: $dir"
  if [[ -d "$dir" ]]; then
   python3 CloudPhinder.py "$dir" --nmin=1 --alpha_crit=5
   else
   echo "directory does not exist $dir"
   fi
   done 
   "processing completed"