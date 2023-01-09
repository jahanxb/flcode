#!/bin/bash


echo "Processing on Client Nodes "
  
ssh jahanxb@10.10.1.2  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 1'

ssh jahanxb@10.10.1.3  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 2'
  
ssh jahanxb@10.10.1.4  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 3'
  
ssh jahanxb@10.10.1.5  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 4'
  
ssh jahanxb@10.10.1.6  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 5'
  
ssh jahanxb@10.10.1.7  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 6'
  
ssh jahanxb@10.10.1.8  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 7'
  
ssh jahanxb@10.10.1.9  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 8'
  
ssh jahanxb@10.10.1.10  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 9'
  
ssh jahanxb@10.10.1.11  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_fed_10_nodes-mongodb-fmnist.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 10'
  
echo 'Process Finished!!!'
