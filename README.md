# Transformer-Enhanced Multimodal Time Series Forecasting via Decoupled Dual-Temporal Graph Adaptation

> >Running style


> >(1) Setting up the experimental task environment: you can do a manual setup of parser.add_argument in scripts (e.g., economy_gnn.sh)

> >1.1 all_models=("Transformer")

> >1.2 --seq_len 8 \
       --label_len 4 \
       --pred_lengths=(12) \
       --use_gnn 1 \  
       --alpha 0 \
    You can select alpha in interval [0,1] for hyperparameters tuning.

> >1.3 Run it directly from the command line：nohup bash ./scripts/economy_gnn.sh 0 0 0 > try.log 2>&1 &

> > The results are in the corresponding train.log file.