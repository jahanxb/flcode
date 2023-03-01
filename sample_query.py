from pymongo import MongoClient
from neo4j import GraphDatabase
from psycopg2 import connect
import pika
from cassandra.cluster import Cluster
import time


def mongo_query():
    # MongoDB query
    mongodb_url = 'mongodb://jahanxb:phdunr@10.10.1.1:27017/iteration_status?authMechanism=DEFAULT&authSource=admin&tls=false'

    mongo_client = MongoClient(mongodb_url)
    db = mongo_client.iteration_status
    start_time = time.time()
    result = db.master_global.find({'task_id': 'master_global_for_node[1]_round[0]'})
    #result = db.master_global.find_one()

    # for doc in result:
    #     print('data: ', doc)
    
    end_time = time.time()
    mongo_query_time = (end_time - start_time) * 1000
    print("MongoDB query time:", mongo_query_time, "ms")


def neo4j_query():
    # Neo4j query
    uri_neo4j = "neo4j://10.10.1.10:7687"
    user_neo4j = "neo4j"
    password_new4j = "oi2KksBMaHfsB355HdoHsI2Kzv4NoOUm7MnPNtnESIY"
    
    neo4j_driver = GraphDatabase.driver(uri_neo4j, auth=(user_neo4j, password_new4j))
    
    #neo4j_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    with neo4j_driver.session() as session:
        start_time = time.time()
        result = session.run("MATCH (n {task_id: 'master_global_for_node[1]_round[0]'}) RETURN n")
        #print(result.data())
        end_time = time.time()
        neo4j_query_time = (end_time - start_time) * 1000
        print("Neo4j query time:", neo4j_query_time ,"ms")
        
        

def postgres_query():
    # PostgreSQL query
    pg_conn = connect(database = "ddfl", user = "postgres", password = "ng.dB.Q'3s`^9HVx", host = "35.224.200.63", port = "5432")
    with pg_conn.cursor() as cur:
        start_time = time.time()
        cur.execute("SELECT * FROM iteration_status.master_global WHERE task_id = 'master_global_for_node[5]_round[0]';")
        result = cur.fetchall()
        #print(result)
        end_time = time.time()
        pg_query_time = (end_time - start_time) * 1000
        print("PostgreSQL query time:", pg_query_time, "ms")

# # RabbitMQ query
# connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
# channel = connection.channel()
# channel.queue_declare(queue='test_queue')
# start_time = time.time()
# method, properties, body = channel.basic_get(queue='test_queue', auto_ack=True)
# end_time = time.time()
# rabbitmq_query_time = end_time - start_time
# print("RabbitMQ query time:", rabbitmq_query_time)

def cassandra_query():
    # Cassandra query
    cluster = Cluster(['10.10.1.2'])
    session = cluster.connect('iteration_status')
    start_time = time.time()
    result = session.execute("SELECT * FROM master_global WHERE task_id = 'master_global_for_node[5]_round[0]'")
    result = result.one()
    #print(result)
    end_time = time.time()
    cassandra_query_time = (end_time - start_time) * 1000
    print("Cassandra query time:", cassandra_query_time, "ms")
    

def scp_query():
    import subprocess
    import time

    # Set the source and destination file paths for the file transfer
    src_path = '/mydata/flcode/models/nodes_trained_model/global_models/master_global_for_node[1]_round[0].pkl'
    dst_path = 'jahanxb@10.10.1.2:/users/jahanxb/master_global_for_node[1]_round[0].pkl'

    # Run the scp command and capture the output
    start_time = time.time()
    process = subprocess.Popen(['scp', src_path, dst_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()
    

    # Print the output and the time taken by the command
    print(f'STDOUT: {stdout.decode()}')
    print(f'STDERR: {stderr.decode()}')
    print(f'Time taken: {(end_time - start_time) * 1000} ms')

if __name__ == '__main__':
    mongo_query()
    #postgres_query()
    #cassandra_query()
    #scp_query()
    #neo4j_query()
