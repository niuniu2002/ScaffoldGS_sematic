#!/usr/bin/env bash
set -e

# ============================================================
# 批量实验脚本：串行运行 6 个参数微调实验
# 说明：不动代码，只改命令行参数。已完成的实验自动跳过；
#      若检测到其他 train.py 在跑，会等待其结束后再启动下一个。
# ============================================================

echo "=========================================="
echo "开始批量实验：$(date)"
echo "=========================================="

wait_for_other_train() {
    while pgrep -f "train.py -s data/dronev4_2" > /dev/null; do
        echo "[WAIT] 检测到其他 train.py 正在运行，等待 60 秒..."
        sleep 60
    done
}

# 实验列表与对应输出目录
declare -A EXP_MAP=(
    ["run_exp11_focalalpha75_fromscratch.sh"]="output/dronev4_2_exp11_focalalpha75_fromscratch"
    ["run_exp12_maskweight015_fromscratch.sh"]="output/dronev4_2_exp12_maskweight015_fromscratch"
    ["run_exp13_maskweight025_fromscratch.sh"]="output/dronev4_2_exp13_maskweight025_fromscratch"
    ["run_exp14_updateuntil20000_fromscratch.sh"]="output/dronev4_2_exp14_updateuntil20000_fromscratch"
    ["run_exp15_startsem3000_mask02.sh"]="output/dronev4_2_exp15_startsem3000_mask02"
    ["run_exp16_semstart0_fromscratch.sh"]="output/dronev4_2_exp16_semstart0_fromscratch"
)

for exp in "${!EXP_MAP[@]}"; do
    out_dir="${EXP_MAP[$exp]}"

    # 如果输出目录已存在且包含 results.json，视为已完成，自动跳过
    if [ -f "$out_dir/results.json" ]; then
        echo ""
        echo "[SKIP] $exp 已完成（找到 $out_dir/results.json）"
        continue
    fi

    # 如果输出目录已存在且包含 outputs.log，说明正在跑或之前中断，也跳过
    if [ -f "$out_dir/outputs.log" ]; then
        echo ""
        echo "[SKIP] $exp 已有 outputs.log，视为已运行或进行中"
        continue
    fi

    # 等待其他训练任务结束
    wait_for_other_train

    echo ""
    echo ">>> 启动实验: $exp"
    echo ">>> 当前时间: $(date)"
    echo ""
    bash "autoresearch_scaffoldgs/$exp"
    echo ""
    echo "<<< 实验完成: $exp"
    echo "<<< 完成时间: $(date)"
    echo ""
done

echo "=========================================="
echo "全部实验完成：$(date)"
echo "=========================================="
