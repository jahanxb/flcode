
# from sys import api_version
# from kafka import KafkaProducer

# import time
# import json
# from datetime import datetime

# def kf():
#     #producer = KafkaProducer(boostrap_servers='localhost:9092')
    
#     producer = KafkaProducer(
#         value_serializer=lambda m: json.dumps(m).encode('UTF-8')
#         ,bootstrap_servers=['10.10.1.3:39092'])
    
#     for i in range(1,100):
#         producer.send('t1',value={"hello":i})
#         time.sleep(0.02)
    
#     #future = producer.send('youtube',b'hello')
#     #result = future.get(timeout=10)
#     #producer.flush()



# kf()
import json

from sys import api_version
from kafka import KafkaProducer
from kafka.errors import KafkaError
import msgpack
producer = KafkaProducer(bootstrap_servers=['10.10.1.3:9092'],api_version=(0,11,5))

# Asynchronous by default
future = producer.send('my-topic', b'raw_bytes')

# Block for 'synchronous' sends
try:
    record_metadata = future.get(timeout=10)
except KafkaError:
    # Decide what to do if produce request failed...
    log.exception()
    pass

# Successful result returns assigned partition and offset
print (record_metadata.topic)
print (record_metadata.partition)
print (record_metadata.offset)

# produce keyed messages to enable hashed partitioning
producer.send('my-topic', key=b'foo', value=b'bar')

# encode objects via msgpack
producer = KafkaProducer(value_serializer=msgpack.dumps)
producer.send('msgpack-topic', {'key': 'value'})

# produce json messages
producer = KafkaProducer(value_serializer=lambda m: json.dumps(m).encode('ascii'))
producer.send('json-topic', {'key': 'value'})

# produce asynchronously
for _ in range(100):
    producer.send('my-topic', b'msg')

def on_send_success(record_metadata):
    print(record_metadata.topic)
    print(record_metadata.partition)
    print(record_metadata.offset)

def on_send_error(excp):
    log.error('I am an errback', exc_info=excp)
    # handle exception

# produce asynchronously with callbacks
producer.send('my-topic', b'raw_bytes').add_callback(on_send_success).add_errback(on_send_error)

# block until all async messages are sent
producer.flush()

# configure multiple retries
producer = KafkaProducer(retries=5)