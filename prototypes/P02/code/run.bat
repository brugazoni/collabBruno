@echo off
python preprocess.py %1
python estimate.py %1
