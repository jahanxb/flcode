#!/bin/bash


scp -r cassandra_configs/cassandra-node1.yaml jahanxb@10.10.1.2:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node2.yaml jahanxb@10.10.1.3:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node3.yaml jahanxb@10.10.1.4:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node4.yaml jahanxb@10.10.1.5:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node5.yaml jahanxb@10.10.1.6:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node6.yaml jahanxb@10.10.1.7:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node7.yaml jahanxb@10.10.1.8:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node8.yaml jahanxb@10.10.1.9:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node9.yaml jahanxb@10.10.1.10:/users/jahanxb/cassandra.yaml.1
scp -r cassandra_configs/cassandra-node10.yaml jahanxb@10.10.1.11:/users/jahanxb/cassandra.yaml.1

ssh jahanxb@10.10.1.2 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.3 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.4 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.5 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.6 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.7 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.8 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.9 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.10 -f "sudo systemctl stop cassandra"
ssh jahanxb@10.10.1.11 -f "sudo systemctl stop cassandra"


ssh jahanxb@10.10.1.2 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.3 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.4 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.5 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.6 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.7 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.8 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.9 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.10 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
ssh jahanxb@10.10.1.11 -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"



ssh jahanxb@10.10.1.2 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.3 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.4 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.5 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.6 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.7 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.8 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.9 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.10 -f "sudo rm -rf /var/log/cassandra/*"
ssh jahanxb@10.10.1.11 -f "sudo rm -rf /var/log/cassandra/*"


ssh jahanxb@10.10.1.2 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.3 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.4 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.5 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.6 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.7 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.8 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.9 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.10 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
ssh jahanxb@10.10.1.11 -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"



ssh jahanxb@10.10.1.2 -f "sudo systemctl start cassandra"
ssh jahanxb@10.10.1.4 -f "sudo systemctl start cassandra"
ssh jahanxb@10.10.1.5 -f "sudo systemctl start cassandra"
ssh jahanxb@10.10.1.6 -f "sudo systemctl start cassandra"
ssh jahanxb@10.10.1.7 -f "sudo systemctl start cassandra"
ssh jahanxb@10.10.1.8 -f "sudo systemctl start cassandra"
ssh jahanxb@10.10.1.9 -f "sudo systemctl start cassandra"
ssh jahanxb@10.10.1.10 -f "sudo systemctl start cassandra"
ssh jahanxb@10.10.1.11 -f "sudo systemctl start cassandra"


# x=2
# while [ $x -le 11 ]
# do
#   echo "Welcome $x times"
#   ssh jahanxb@10.10.1.$x -f "sudo systemctl stop cassandra"
#   ssh jahanxb@10.10.1.$x -f "sudo rm -rf /var/lib/cassandra/data/iteration_status/*"
#   ssh jahanxb@10.10.1.$x -f "sudo rm -rf /var/log/cassandra/*"
#   ssh jahanxb@10.10.1.$x -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
#   #ssh jahanxb@10.10.1.$x -f "sudo mv /etc/cassandra/cassandra.yaml /etc/cassandra/cassandra.yaml.bck"
#   ##scp -r cassandra.yaml jahanxb@10.10.1.$x:/users/jahanxb/cassandra.yaml.1
#   #ssh jahanxb@10.10.1.$x -f "sudo cp /users/jahanxb/cassandra.yaml.1 /etc/cassandra/cassandra.yaml"
#   #ssh jahanxb@10.10.1.$x -f "sudo rm -rf /var/lib/cassandra/*"
#   ssh jahanxb@10.10.1.$x -f "sudo systemctl start cassandra"

#   x=$(( $x + 1 ))
# done
