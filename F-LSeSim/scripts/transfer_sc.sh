echo "Read config file: $1";
mode=$(grep -A0 'MODEL_NAME:' $1 | cut -c 13- | cut -d# -f1 | tr -d '"' | sed "s/ //g"); echo "$mode"
if [ "$mode" = "LSeSim" ]; then
    exp=$(grep -A0 'EXPERIMENT_NAME:' $1 | cut -c 18- | tr -d '"')
    python3 inference.py \
    --name $exp \
    --model sc \
    -c $1

    python3 combine.py \
    --read_original \
    -c $1
else
    echo "Not LSeSim model"
fi
