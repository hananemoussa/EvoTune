#!/bin/bash
# Quick monitoring script for EvoTune experiments

echo "========================================="
echo "  EvoTune Experiments Monitor"
echo "========================================="
echo ""

echo "📊 Job Status:"
squeue -u $USER -o "%.10i %.12P %.30j %.2t %.10M %.6D %R"
echo ""

echo "📁 Recent Log Files:"
ls -lth logs/*.out 2>/dev/null | head -5
echo ""

echo "🔍 Latest Progress (Baseline - Job 2916473):"
grep "ROUND.*FINISHED\|Best overall program score" logs/funsearch_llama_1B_baseline_2916473.out 2>/dev/null | tail -5
echo ""

echo "🔍 Latest Progress (EvoTune - Job 2916475):"
grep "ROUND.*FINISHED\|Best overall program score\|TRAINING MODEL" logs/evotune_llama_1B_2916475.out 2>/dev/null | tail -5
echo ""

echo "⚠️  Recent Errors (if any):"
tail -10 logs/*.err 2>/dev/null | grep -i "error\|exception\|failed" || echo "No recent errors found"
echo ""

echo "========================================="
echo "Commands:"
echo "  - Watch live: watch -n 5 squeue -u \$USER"
echo "  - Tail baseline: tail -f logs/funsearch_llama_1B_baseline_2916473.out"
echo "  - Tail evotune: tail -f logs/evotune_llama_1B_2916475.out"
echo "  - Cancel jobs: scancel 2916473 2916475"
echo "========================================="
