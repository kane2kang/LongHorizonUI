#!/bin/bash

uv venv --python 3.11
source .venv/bin/activate
uv pip install pyside6~=6.6.3.1
uv pip install git+https://github.com/leng-yue/py-scrcpy-client
uv pip install numpy~=1.26
uv pip install setuptools
uv pip install pip
uv pip install adbutils~=1.2
python main.py