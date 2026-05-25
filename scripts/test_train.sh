python3 heat-pinn-cli/heat.py train \
    --domain data/geometry/square/domain_data.parquet \
    --boundary data/geometry/square/boundary_data.parquet \
    --model test_model.joblib \
    --epochs 20 \
    --every 2