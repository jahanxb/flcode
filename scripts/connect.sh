
#!/bin/bash

rm -rf /mydata/flcode/models/nodes_trained_model/global_models/* && rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/* && rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*
#scp -r cassandra.yaml jahanxb@10.10.1.2:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.3:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.4:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.5:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.6:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.7:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.8:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.9:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.10:/users/jahanxb/cassandra.yaml.1
#scp -r cassandra.yaml jahanxb@10.10.1.11:/users/jahanxb/cassandra.yaml.1

x=2
while [ $x -le 11 ]
do
  echo "Welcome $x times"
  ssh jahanxb@10.10.1.$x -f 'killall screen'
  ssh jahanxb@10.10.1.$x -f "screen -ls | grep '(Detached)' | awk '{print $1}' | xargs -I % -t screen -X -S % quit"
  
  #ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model"
  
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/global_models"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/nodes_local"
  ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_trained_model/nodes_local_loss"

  #ssh jahanxb@10.10.1.$x -f "sudo chown -R jahanxb:root /mydata/flcode/models/nodes_trained_model"
  #ssh jahanxb@10.10.1.$x -f "sudo chown -R jahanxb:root /mydata/flcode/models/node_output"
  #ssh jahanxb@10.10.1.$x -f "cd /mydata/flcode/ && git stash && git pull"
  #ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_sftp"
  #ssh jahanxb@10.10.1.$x -f "mkdir /mydata/flcode/models/nodes_sftp/global_models/* && mkdir /mydata/flcode/models/nodes_sftp/nodes_local/* && mkdir /mydata/flcode/models/nodes_sftp/nodes_local_loss/*"
  #ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_sftp/global_models/* && sudo rm -rf /mydata/flcode/models/nodes_sftp/nodes_local/* && sudo rm -rf /mydata/flcode/models/nodes_sftp/nodes_local_loss/*"
  #ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/global_models/* && sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/* && sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/global_models/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/*"
  ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*"
  #ssh jahanxb@10.10.1.$x -f "sudo rm -rf /mydata/flcode/node_output"
  
  #ssh jahanxb@10.10.1.$x -f "sudo systemctl stop cassandra"
  #ssh jahanxb@10.10.1.$x -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
  #ssh jahanxb@10.10.1.$x -f "sudo rm -rf /var/log/cassandra/*"

  #ssh jahanxb@10.10.1.$x -f "sudo mv /etc/cassandra/cassandra.yaml /etc/cassandra/cassandra.yaml.bck"
  ##scp -r cassandra.yaml jahanxb@10.10.1.$x:/users/jahanxb/cassandra.yaml.1
  #ssh jahanxb@10.10.1.$x -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
  #ssh jahanxb@10.10.1.$x -f "sudo rm -rf /var/lib/cassandra/*"
  #ssh jahanxb@10.10.1.$x -f "sudo systemctl start cassandra"



  #ssh jahanxb@10.10.1.$x -f 'rm -rf /mydata/flcode/global_models && rm -rf /mydata/flcode/nodes_local && rm -rf /mydata/flcode/nodes_local_loss'
  
  #scp options.py jahanxb@10.10.1.$x:/mydata/flcode/options.py
  x=$(( $x + 1 ))
done
