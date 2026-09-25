@echo off
cd /d C:\Users\AbhiDS\Documents\AmazonMLChallenge26
echo [1/3] recall gate on 5k ...
python code\business_entity_resolution\src\blocking.py --n 5000 --min-recall 0.97 > logs\check5k.log 2>&1
if errorlevel 1 (
    echo GATE FAILED - see logs\check5k.log
    exit /b 1
)
echo [2/3] rebuild pairs on 100k ...
python code\business_entity_resolution\src\build_train_pairs.py --n 100000 > logs\rebuild.log 2>&1
if errorlevel 1 (
    echo REBUILD FAILED - see logs\rebuild.log
    exit /b 1
)
echo [3/3] retrain ...
python code\business_entity_resolution\src\train_model.py > logs\retrain.log 2>&1
echo DONE