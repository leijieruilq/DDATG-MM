# Transformer-Enhanced Multimodal Time Series Forecasting via Decoupled Dual-Temporal Graph Adaptation

## Running style


> >(1) Setting up the experimental task environment: you can see the shell scripts tailored to the main Transformer-based backbones (e.g., Transformer, Informer, Autoformer, Nonstationary, PatchTST and iTransformer) in "scripts" folder.

> >(2) Please noticed that ** *_gnn.sh ** is implemented with our proposed GNN plugin Decoupled Dual Adaptive Temporal Graph (DDATG), for details you can find in the 2.1, 2.2 below:

> >2.1 all_models=("Transformer")

> >2.2 --seq_len 8 \
       --label_len 4 \
       --pred_lengths=(12) \
       --use_gnn 1 \  # use our DDATG \
       --alpha 0 \ # regulatory factor for adjusting the strength (Joint Alignment Training Loss in Time-Frequency domain) \
    You can select alpha in interval [0,1] for hyperparameters tuning.

> >2.3 Run it directly from the command line：nohup bash ./scripts/Transformer/algriculture_gnn.sh 0 0 0 > train.log 2>&1 &

> > The log results are in the corresponding train.log file.
