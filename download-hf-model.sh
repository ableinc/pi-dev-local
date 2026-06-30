#!/usr/bin/env bash

# hf download unsloth/gemma-4-12B-it-qat-GGUF --include "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf" --include "mmproj-F16.gguf" --include "mtp-gemma-4-12B-it.gguf" --local-dir /mnt/storage/

PROVIDER="$1"
FILES="$2"
DOWNLOAD_PATH="$3"

usage () {
   echo -e "Usage: download-hf-model.sh <provider> file1,file2 download-path\n"
}

# Ensure provider is not empty
if [ -z "$PROVIDER" ]; then
   usage
   exit 1
fi

# Ensure files is not empty
if [ -z "$FILES" ]; then
   usage
   exit 1
fi

# Ensure download path is not empty
if [ -z "$DOWNLOAD_PATH" ]; then
   usage
   exit 1
fi

if ! [ -d "$DOWNLOAD_PATH" ]; then
  echo "$DOWNLOAD_PATH is not a directory or does not exist"
  exit 1
fi

# Split files by delimiter and into a new array
echo -e "Downloading these files: $FILES\n"

# Use a local IFS for splitting to avoid affecting other parts of the script
IFS=',' read -ra files_array <<< "$FILES"

# Create command
CMD="hf download $PROVIDER "
for i in "${files_array[@]}"; do
     CMD+="--include $i "
done
# Append local download path to command
CMD+="--local-dir $DOWNLOAD_PATH"

echo "Executing: $CMD"

# Execute command
$CMD
