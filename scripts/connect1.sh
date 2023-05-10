#!/bin/bash
x=2
while [ $x -le 11 ]
do
  echo "Welcome $x times"
  ssh jahanxb@10.10.1.$x -f 'killall screen'
  ssh jahanxb@10.10.1.$x -f "screen -ls | grep '(Detached)' | awk '{print $1}' | xargs -I % -t screen -X -S % quit"
  
  ssh jahanxb@10.10.1.$x -f 'cd /mydata/flcode && git stash && git pull origin main_local_clients --force'
  
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/global_models"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/nodes_local"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/nodes_local_loss"

  #ssh jahanxb@10.10.1.$x -f "sudo chown -R jahanxb:root /mydata/flcode/models/nodes_trained_model"
  #ssh jahanxb@10.10.1.$x -f "sudo chown -R jahanxb:root /mydata/flcode/models/node_output"
  #ssh jahanxb@10.10.1.$x -f "cd /mydata/flcode/ && git stash && git pull"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_sftp"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_sftp/global_models"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_sftp/nodes_local"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_sftp/nodes_local_loss/"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_sftp/global_models/* && sudo rm -rf /mydata/flcode/models/nodes_sftp/nodes_local/* && sudo rm -rf /mydata/flcode/models/nodes_sftp/nodes_local_loss/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/global_models/* && sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/* && sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/global_models/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*"
  
  x=$(( $x + 1 ))
done
