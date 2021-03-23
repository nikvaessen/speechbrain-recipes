set -e

# print debugging information
echo "pwd: $(pwd)"
echo "experiment: $(pwd)/$1"
echo "DEBUG: $DEBUG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
poetry debug

# run the desired experiment file
"$(pwd)"/"$1"