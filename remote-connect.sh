#!/bin/bash
# echo "Initiate Global Model Process on Master Node"
# /mydata/flcode/venv/bin/python main_global_server.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 10 --frac 1 --db scp &

echo "Processing on Client Nodes "
echo "Local Node Process Initiated on Node 1 [IP: 10.10.1.2]" 
ssh jahanxb@10.10.1.2  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 1 --frac 1 --db scp' 
echo "Local Node Process Initiated on Node 2 [IP: 10.10.1.3]"
ssh jahanxb@10.10.1.3  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 2 --frac 1 --db scp' 
echo "Local Node Process Initiated on Node 3 [IP: 10.10.1.4]"
ssh jahanxb@10.10.1.4  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 3 --frac 1 --db scp' 
echo "Local Node Process Initiated on Node 4 [IP: 10.10.1.5]"  
ssh jahanxb@10.10.1.5  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 4 --frac 1 --db scp' 
echo "Local Node Process Initiated on Node 5 [IP: 10.10.1.6]"
ssh jahanxb@10.10.1.6  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 5 --frac 1 --db scp'
echo "Local Node Process Initiated on Node 6 [IP: 10.10.1.7]"
ssh jahanxb@10.10.1.7  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 6 --frac 1 --db scp'
echo "Local Node Process Initiated on Node 7 [IP: 10.10.1.8]"  
ssh jahanxb@10.10.1.8  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 7 --frac 1 --db scp'
echo "Local Node Process Initiated on Node 8 [IP: 10.10.1.9]"
ssh jahanxb@10.10.1.9  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 8 --frac 1 --db scp'
echo "Local Node Process Initiated on Node 9 [IP: 10.10.1.10]"
ssh jahanxb@10.10.1.10  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 9 --frac 1 --db scp'
echo "Local Node Process Initiated on Node 10 [IP: 10.10.1.11]"
ssh jahanxb@10.10.1.11  -f 'screen -dmS sudo /mydata/flcode/venv/bin/python /mydata/flcode/main_local_clients.py --dataset fmnist --round 10 --gpu -1 --tau 10 --num_users 10 --frac 1 --db scp'
  
echo 'Process Finished!!!'
