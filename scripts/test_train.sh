python3 heat-pinn-cli/heat.py train \
    --domain data/square/domain_data.parquet \
    --boundary data/square/boundary_data.parquet \
    --model data/PINN-DNN-2-1-1-10-50802cd8-c014-4b85-9465-815580a225f1.joblib \
    --epochs 20 \
    --every 2