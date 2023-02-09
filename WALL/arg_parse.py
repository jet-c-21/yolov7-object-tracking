# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2023-02-10
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--classes', nargs='+', type=int)
opt = parser.parse_args()
print(opt)
