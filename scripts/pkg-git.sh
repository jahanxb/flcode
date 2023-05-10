#!/bin/bash
echo "Welcome"
rm -rf /mydata/flcode
cd /mydata/ && git clone -b client_10_nodes_redo https://personaltokenhere@github.com/jahanxb/flcode.git
python3 -m venv /mydata/flcode/venv && source /mydata/flcode/venv/bin/activate.csh && /mydata/flcode/venv/bin/pip install --upgrade pip  && /mydata/flcode/venv/bin/pip install wheel && /mydata/flcode/venv/bin/pip install -r /mydata/flcode/reqpy38.txt
