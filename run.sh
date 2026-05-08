#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p ./log/checkpoints

# 定义日志文件路径
LOG_DIR="./log/checkpoints"

# 执行命令并记录日志
echo "开始训练教师模型..."
python -m scripts.train_teacher > "$LOG_DIR/train_teachers.log" 2>&1 &

# 等待上一个命令完成
wait

echo "生成伪标签..."
python -m scripts.gen_pseudo_labels > "$LOG_DIR/gen_pseudo_labels.log" 2>&1 &

# 等待上一个命令完成
wait

echo "训练CPS学生模型..."
python -m scripts.train_cps_student > "$LOG_DIR/train_cps_student.log" 2>&1 &

# 等待上一个命令完成
wait

echo "导出ONNX模型..."
python -m scripts.export_onnx > "$LOG_DIR/export_onnx.log" 2>&1 &

# 等待所有后台任务完成
wait

echo "所有训练任务已完成！"