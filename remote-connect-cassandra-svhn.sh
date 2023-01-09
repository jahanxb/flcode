#!/bin/bash


echo "Processing on Client Nodes "
  
ssh jahanxb@10.10.1.2  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 1 --frac 1 >> /users/jahanxb/node1.txt'
  
ssh jahanxb@10.10.1.3  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 2 --frac 1 >> /users/jahanxb/node2.txt'
  
ssh jahanxb@10.10.1.4  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 3 --frac 1 >> /users/jahanxb/node3.txt'
  
ssh jahanxb@10.10.1.5  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 4 --frac 1 >> /users/jahanxb/node4.txt'
  
ssh jahanxb@10.10.1.6  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 5 --frac 1 >> /users/jahanxb/node5.txt'
  
ssh jahanxb@10.10.1.7  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 6 --frac 1 >> /users/jahanxb/node6.txt'
  
ssh jahanxb@10.10.1.8  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 7 --frac 1 >> /users/jahanxb/node7.txt'
  
ssh jahanxb@10.10.1.9  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 8 --frac 1 >> /users/jahanxb/node8.txt'
  
ssh jahanxb@10.10.1.10  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 9 --frac 1 >> /users/jahanxb/node9.txt'
  
ssh jahanxb@10.10.1.11  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-cassandra-svhn.py --dataset svhn --round 10 --gpu -1 --tau 10 --num_users 10 --frac 1 >> /users/jahanxb/node10.txt'
  
echo 'Process Finished!!!'
