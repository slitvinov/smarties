import sys
import socket
import os, os.path
import time
import numpy as np

sockid = sys.argv[1]
server_address = "/tmp/smarties_sock_"+str(sockid)

try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise

server = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(server_address)

server.listen(1)
conn, addr = server.accept()
print( 'Connected by', addr )
 
while True:
   state = np.zeros([7, 1], dtype=np.float64)
   conn.send(state.tobytes())
   buf = conn.recv(1*8)
   action = np.frombuffer(buf, dtype=np.float64)
   print(action)   




