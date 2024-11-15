# Import packages for TCP server
import socket


def set_trigger(value, connection):
    a = value
    connection.send(bytes(chr(a), 'utf-8'))

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('localhost', 30000) #server address: localhost, port: 30'000
print("starting")
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()
    break


set_trigger(55, connection);
