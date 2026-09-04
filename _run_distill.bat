@echo off
cd /D D:\training
python -u distill.py > D:\training\distill.log 2>&1
echo DONE >> D:\training\distill.log
