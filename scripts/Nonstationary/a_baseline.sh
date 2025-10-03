#!/bin/bash
# 进入脚本所在目录
cd /data/ljr/keyan/MM-TSFlib-main/scripts/Nonstationary

# 运行所有 .sh 文件，但排除 _gnn.sh 结尾的
for script in *.sh; do
    if [[ $script != *_gnn.sh ]]; then
        echo "正在运行: $script"
        bash "$script"
    fi
done