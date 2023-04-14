#!/bin/bash

x=3
while [ $x -le 11 ]
do
  echo "Welcome $x times"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode"
  ssh jahanxb@10.10.1.$x -f "sudo chown -R jahanxb:dsdm-PG0 /mydata"
  

  #ssh jahanxb@10.10.1.$x -f 'cd /mydata/flcode && git stash && git pull origin main_local_clients --force'
  #Removed the github token https://github_token@github.com?
  ssh jahanxb@10.10.1.$x -f "cd /mydata/ && git clone -b main_local_clients https://github.com/jahanxb/flcode.git"
  #ssh jahanxb@10.10.1.$x -f "python3 -m venv /mydata/flcode/venv && source /mydata/flcode/venv/bin/activate.csh && /mydata/flcode/venv/bin/pip install --upgrade pip  && /mydata/flcode/venv/bin/pip install wheel && /mydata/flcode/venv/bin/pip install -r /mydata/flcode/reqpy38.txt"
  sleep 10
  #ssh jahanxb@10.10.1.$x -f "/mydata/flcode/venv/bin/pip install -r /mydata/flcode/reqpy38.txt"


  #scp pkg-git.sh jahanxb@10.10.1.$x:/users/jahanxb/pkg-git.sh
  #ssh jahanxb@10.10.1.$x -f "chmod +x /users/jahanxb/pkg-git.sh"
  #ssh jahanxb@10.10.1.$x -f "./pkg-git.sh"
  #ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model"
  #ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/global_models"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/nodes_local"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/nodes_local_loss"


  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/global_models/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*"

  x=$(( $x + 1 ))
done
