import socket
import numpy as np

class Socket():
    def __init__(self, host_id, port):
        self.host_id = host_id
        self.port = port 
        del host_id
        del port
        self.s = socket.socket()
        self.s.bind((self.host_id,self.port))


    def Listen(self, num_client):
        self.s.listen(num_client)
        print("Waiting for any incoming connections...")
        self.conn, addr = self.s.accept()
        print(addr,"Has connected to the server")


    def Received(self, size_buff):
        buf = b''
        for i in range(15):
            buf +=  self.conn.recv(1024)

        fileoopen = open("san.txt","wb")     
        fileoopen.write(buf)
        fileoopen.close()
        print("Data has been transmitted")

    def Send(self, data):
        self.conn.send(data)


# host_id = '10.42.0.1'
# port = 8080
# soc = Socket(host_id,port)
# soc.Listen(1)
# soc.Received(1024)