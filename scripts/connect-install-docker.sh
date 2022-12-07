
#!/bin/bash

rm -rf /mydata/flcode/models/nodes_sftp/global_models/* && rm -rf /mydata/flcode/models/nodes_sftp/nodes_local/* && rm -rf /mydata/flcode/models/nodes_sftp/nodes_local_loss/*

x=2
while [ $x -le 11 ]
do
  echo "Welcome $x times"
  ssh jahanxb@10.10.1.$x -f 'sudo apt-get update'
  ssh jahanxb@10.10.1.$x -f "sudo apt-get install ca-certificates -y && sudo apt-get install -y curl && sudo apt-get -y install gnupg && sudo apt-get install -y lsb-release"
  ssh jahanxb@10.10.1.$x -f "sudo mkdir -p /etc/apt/keyrings"
  ssh jahanxb@10.10.1.$x -f "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg"
  ssh jahanxb@10.10.1.$x -f "echo 'deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable' |sudo tee /etc/apt/sources.list.d/docker.list > /dev/null "
  ssh jahanxb@10.10.1.$x -f "sudo apt-get install docker-ce -y && sudo apt-get install docker-ce-cli -y && sudo apt-get install -y containerd.io && sudo apt-get install -y docker-compose-plugin && sudo apt-get install docker-compose -y"
  #ssh jahanxb@10.10.1.$x -f 'rm -rf /mydata/flcode/global_models && rm -rf /mydata/flcode/nodes_local && rm -rf /mydata/flcode/nodes_local_loss'
  
  #scp options.py jahanxb@10.10.1.$x:/mydata/flcode/options.py
  x=$(( $x + 1 ))
done
