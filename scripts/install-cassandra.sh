#!/bin/bash

x=2
while [ $x -le 11 ]
do
  echo "Welcome $x times"
  #ssh jahanxb@10.10.1.$x -f 'sudo apt-get update -y && sudo apt install default-jdk -y && sudo apt install wget -y'
  #ssh jahanxb@10.10.1.$x -f "wget -q -O - https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add - "
  #ssh jahanxb@10.10.1.$x -f "echo "deb http://www.apache.org/dist/cassandra/debian 40x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list deb http://www.apache.org/dist/cassandra/debian 40x main "
  
  #scp pkg-cassandra.sh jahanxb@10.10.1.$x:/users/jahanxb/pkg-cassandra.sh
  #ssh jahanxb@10.10.1.$x -f "chmod +x /users/jahanxb/pkg-cassandra.sh"
  #ssh jahanxb@10.10.1.$x -f "./pkg-cassandra.sh"
  #ssh jahanxb@10.10.1.$x -f "sudo apt-get update -y && sudo apt install cassandra -y"
  #ssh jahanxb@10.10.1.$x -f "sudo systemctl enable cassandra && sudo systemctl start cassandra && sudo systemctl status cassandra "
  
  #ssh jahanxb@10.10.1.$x -f "sudo cp /etc/cassandra/cassandra.yaml /etc/cassandra/cassandra.yaml.bck"
  #scp /mydata/flcode/scripts/cassandra.yaml jahanxb@10.10.1.$x:/users/jahanxb/cassandra.yaml
  #ssh jahanxb@10.10.1.$x -f "sudo cp /users/jahanxb/cassandra.yaml /etc/cassandra/cassandra.yaml"
  #ssh jahanxb@10.10.1.$x -f "screen -dmS sudo systemctl stop cassandra"
  #ssh jahanxb@10.10.1.$x -f "screen -dmS sudo systemctl start cassandra"

  scp /mydata/flcode/scripts/cassandra-sys.sh jahanxb@10.10.1.$x:/users/jahanxb/cassandra-sys.sh
  ssh jahanxb@10.10.1.$x -f "chmod +x /users/jahanxb/cassandra-sys.sh"
  ssh jahanxb@10.10.1.$x -f "./cassandra-sys.sh"
  sleep 5


  x=$(( $x + 1 ))
done
