mkdir -p validation

GEOM_PATH="validation"
MODEL_NAME="validation/model_000.joblib"
INFER_OUT="validation/inference_result.parquet"

echo "[*] BUILD GEOMETRY"
python3 utils/square.py \
  --data-per-boundary 25 \
  --Nc 10000 \
  --output-path "${GEOM_PATH}" \
  --bc-values 0 0.5 0.3 1

echo "[*] BUILD MODEL"
python3 heat-pinn-cli/heat.py build \
    --in-shape 2 \
    --out-shape 1 \
    --n-hidden-layers 1 \
    --neuron-per-layer 10 \
    --actfun tanh \
    --name "${MODEL_NAME}"

echo "[*] TRAIN MODEL"
python3 heat-pinn-cli/heat.py train \
    --domain "${GEOM_PATH}/domain_data.parquet" \
    --boundary "${GEOM_PATH}/boundary_data.parquet" \
    --model "${MODEL_NAME}" \
    --epochs 1000 \
    --every 100

echo "[*] MODEL INFERENCE"
python3 heat-pinn-cli/heat.py infer \
    --data "${GEOM_PATH}/domain_data.parquet" \
    --model "${MODEL_NAME}" \
    --output "${INFER_OUT}"