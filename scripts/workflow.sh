mkdir -p validation/rod

GEOM_PATH="data/geometry/rod"
MODEL_NAME="validation/rod/model_000.joblib"
INFER_OUT="validation/rod/inference_result.parquet"

echo "[*] BUILD MODEL"
python3 heat-pinn-cli/heat.py build \
    --in-shape 2 \
    --out-shape 1 \
    --n-hidden-layers 5 \
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