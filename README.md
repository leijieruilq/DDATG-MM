# (IEEE SPL 2026) Transformer-Enhanced Multimodal Time Series Forecasting via Decoupled Dual-Temporal Graph Adaptation

## Introduction

With the proliferation of multimodal data in real-world applications, integrating time series with auxiliary modalities has become critical for accurate forecasting. Although Transformers and pre-trained language model (PLM) have enabled initial explorations of multi-domain multimodal time series analysis, several pressing challenges still remain. Specifically, coarse-grained alignment may hinder long-range semantic capture, while distribution shifts in intra-modality introduce fluctuating noise. Inspired by GNNs' capability to model spatio-temporal dependencies and contextual interactions, we propose Decoupled Dual Adaptive Temporal Graph (DDATG), a universal GNN plugin for Transformer-PLM based adaptive text-time series bimodal learning. Our framework: (1) Reconstructs global temporal patterns from decoupled local residual terms in temporal modality, enhancing local-global semantic discovery and diversifying attention mechanisms; (2) Explicitly constructs pointwise contextual connections and strengthens aggregation in textual modality, facilitating inter-modal semantic alignment. Extensive experiments across Transformer variants and domain-specific datasets demonstrate the effectiveness of DDATG.
<img width="7748" height="2540" alt="image" src="https://github.com/user-attachments/assets/d03a2386-4342-4736-8c89-44dd7fbee396" />


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

## Citation

If you find this repo helpful, please cite our paper. 

```
@ARTICLE{11230812,
  author={Lei, Jierui and Zhang, Wenjian and Yang, Qingyi and Zhang, Xudong and Tang, Haina},
  journal={IEEE Signal Processing Letters}, 
  title={Transformer-PLM Enhanced Multimodal Time Series Forecasting via Decoupled Dual-Temporal Graph Adaptation}, 
  year={2026},
  volume={33},
  number={},
  pages={11-15},
  keywords={Transformers;Time series analysis;Forecasting;Adaptation models;Semantics;Modeling;Market research;Predictive models;Graph neural networks;Electronic mail;Multimodal time series forecasting;graph structure adaptation;multimodal representation learning},
  doi={10.1109/LSP.2025.3630087}}

```
