#!/bin/bash
curl -svL $1 | tar -xz
cd tensorrt_llm
python3 scripts/build_wheel.py --use_ccache -j 32 --trt_root /usr/local/tensorrt -a '90-real'
