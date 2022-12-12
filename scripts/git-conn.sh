#!/bin/bash

x=3
while [ $x -le 11 ]
do
  echo "Welcome $x times"
  ssh jahanxb@10.10.1.$x -f 'cd /mydata/flcode && git stash && git pull origin client_10_nodes_encrypt_decentralized --force'
  #ssh jahanxb@10.10.1.$x -f "cd /mydata/ && git clone -b client_10_nodes_redo https://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@github.com/jahanxb/flcode.git"
  #ssh jahanxb@10.10.1.$x -f "python3 -m venv /mydata/flcode/venv && source /mydata/flcode/venv/bin/activate.csh && /mydata/flcode/venv/bin/pip install --upgrade pip  && /mydata/flcode/venv/bin/pip install wheel && /mydata/flcode/venv/bin/pip install -r /mydata/flcode/reqpy38.txt"
  
  #ssh jahanxb@10.10.1.$x -f "/mydata/flcode/venv/bin/pip install -r /mydata/flcode/reqpy38.txt"

  #scp pkg-git.sh jahanxb@10.10.1.$x:/users/jahanxb/pkg-git.sh
  #ssh jahanxb@10.10.1.$x -f "chmod +x /users/jahanxb/pkg-git.sh"
  #ssh jahanxb@10.10.1.$x -f "./pkg-git.sh"

  x=$(( $x + 1 ))
done
