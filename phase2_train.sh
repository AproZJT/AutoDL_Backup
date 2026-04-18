#!/usr/bin/env bash

# 生成 Student Config
python phase2_student_cfg_builder.py

echo "=================================================="
echo "🔥 Phase 2 启动：Teacher-Student 离线蒸馏"
echo "=================================================="
echo "训练已挂入后台，断网不会中断。"
echo "请使用命令查看实时进度: tail -f train_student.log"

# 使用 nohup 启动，并将标准输出和错误一并写入 log
nohup python external_modules/mmdetection/tools/train.py \
    phase2_student_finetune.py \
    > train_student.log 2>&1 &

echo "✅ 炼丹炉已点火！PID: $!"
