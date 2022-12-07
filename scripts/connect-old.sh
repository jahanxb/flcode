#!/bin/bash

rm -rf /mydata/flcode/models/nodes_sftp/global_models/* && rm -rf /mydata/flcode/models/nodes_sftp/nodes_local/* && rm -rf /mydata/flcode/models/nodes_sftp/nodes_local_loss/*

x=2
while [ $x -le 11 ]
do
  echo "Welcome $x times"
  ssh jahanxb@10.10.1.$x -f 'killall screen'
  ssh jahanxb@10.10.1.$x -f "screen -ls | grep '(Detached)' | awk '{print $1}' | xargs -I % -t screen -X -S % quit"
  ssh jahanxb@10.10.1.$x -f "cd /mydata/flcode/ && git stash && git pull"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_sftp"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_sftp/global_models && mkdir /mydata/flcode/models/nodes_sftp/nodes_local/ && mkdir /mydata/flcode/models/nodes_sftp/nodes_local_loss/"
  ssh jahanxb@10.10.1.$x -f "rm -rf /mydata/flcode/models/nodes_sftp/global_models/* && rm -rf /mydata/flcode/models/nodes_sftp/nodes_local/* && rm -rf /mydata/flcode/models/nodes_sftp/nodes_local_loss/*"
  #ssh jahanxb@10.10.1.$x -f 'rm -rf /mydata/flcode/global_models && rm -rf /mydata/flcode/nodes_local && rm -rf /mydata/flcode/nodes_local_loss'
  
  #scp options.py jahanxb@10.10.1.$x:/mydata/flcode/options.py
  x=$(( $x + 1 ))
done
