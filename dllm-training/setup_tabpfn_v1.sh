#!/bin/bash
# Fetch TabPFN v1 (automl/TabPFN, tag v1.0.0: code, prior and released checkpoint) into dllm-training/third_party/,
# plus ConfigSpace (needed only to build TabPFN's prior) into third_party/pydeps. Both are git-ignored.
set -e
cd "$(dirname "$0")"
mkdir -p third_party
if [ ! -d third_party/TabPFN ]; then
  git clone --depth 1 --branch v1.0.0 https://github.com/automl/TabPFN.git third_party/TabPFN
fi
if [ ! -d third_party/pydeps/ConfigSpace ]; then
  python -m pip install --no-deps --target third_party/pydeps "ConfigSpace==1.2.2" more-itertools pyparsing
fi
echo "TabPFN v1 ready in $(pwd)/third_party"
