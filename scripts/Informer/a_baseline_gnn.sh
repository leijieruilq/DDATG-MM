#!/bin/bash
# 进入脚本所在目录
cd /data/ljr/keyan/MM-TSFlib-main/scripts/Informer

# 运行所有 .sh 文件
for script in *_gnn.sh; do
    echo "正在运行: $script"
    bash "$script"
done