*** Please do not share/publish this code *** 

*** environment ***
* ./envfl.yml: python environment (run "conda env create -f envfl.yml" to creat environment)

*** hyperparameter ***
* ./options.py: hyperparameter settings
* ./options_xxx_tuned.py: well-tuned hyperparameters for different datasets (copy to options.py)

*** Example ***
* ./main_fed.py: implementation of the fedavg algorithm (https://arxiv.org/abs/1602.05629)

python main_fed.py 

sudo apt install python3.7
    19  16:44   sudo apt install python3.7-venv
    20  16:44   sudo apt install python3.7-dev

 python main_fed.py --gpu -1 --dataset cifar --round 50
 
 python -m grpc_tools.protoc -I/home/jahanxb/PycharmProjects/FLcode/proto --python_out=. --grpc_python_out=. /home/jahanxb/PycharmProjects/FLcode/proto/pingpong.proto


delete all rabbitmq queues

rabbitmqadmin -u jahanxb -p phdunr -f tsv -q list queues name | while read queue; do rabbitmqadmin -u jahanxb -p phdunr -q delete queue name=${queue}; done

*** Mongodb Setup ***
sudo docker pull mongo:4.0.4
 sudo docker run -d -p 27017:27017 --name test-mongo mongo:latest
 sudo docker exec -it container_name /bin/bash
 mongosh
db. createUser( { user: "jahanxb", pwd: "phdunr", roles: [ { role: "userAdminAnyDatabase", db: "admin" } ] } )

connect to mongoserver through Mongodb Compass 

---------------------------------------



