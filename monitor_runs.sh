#!/bin/bash
# Quick monitoring script for EvoTune experiments
# Configure these variables for your runs:

#===========================================
# CONFIGURATION - Edit these values
#===========================================
PREFIX="sr"                    # Experiment prefix (e.g., "sr", "bin")
FUNSEARCH_JOB="2950253"        # FunSearch baseline job ID
EVOTUNE_JOB="2950255"          # EvoTune DPO job ID
DESCRIPTION="Symbolic Regression - BPG10 (phi)"
#===========================================

echo "========================================="
echo "  EvoTune Experiments Monitor"
echo "========================================="
echo ""

echo "Job Status:"
squeue -u $USER -o "%.10i %.12P %.30j %.2t %.10M %.6D %R"
echo ""

echo "Recent Log Files:"
ls -lth logs/*.out 2>/dev/null | head -8
echo ""

echo "-------------------------------------------"
echo "  ${DESCRIPTION}"
echo "-------------------------------------------"

echo "FunSearch Baseline (Job ${FUNSEARCH_JOB}):"
grep "ROUND.*FINISHED\|Best overall program score\|best_overall_score" logs/*${FUNSEARCH_JOB}.out 2>/dev/null | tail -5
echo ""

echo "EvoTune DPO (Job ${EVOTUNE_JOB}):"
grep "ROUND.*FINISHED\|Best overall program score\|best_overall_score\|TRAINING MODEL" logs/*${EVOTUNE_JOB}.out 2>/dev/null | tail -5
echo ""

echo "Recent Errors (if any):"
tail -10 logs/*${FUNSEARCH_JOB}.err logs/*${EVOTUNE_JOB}.err 2>/dev/null | grep -i "error\|exception\|failed" || echo "No recent errors found"
echo ""

echo "========================================="
echo "Commands:"
echo "  - Watch live: watch -n 5 squeue -u \$USER"
echo ""
echo "  ${PREFIX}:"
echo "    tail -f logs/*${FUNSEARCH_JOB}.out  # FunSearch"
echo "    tail -f logs/*${EVOTUNE_JOB}.out  # EvoTune"
echo ""
echo "  Cancel all: scancel ${FUNSEARCH_JOB} ${EVOTUNE_JOB}"
echo "========================================="
