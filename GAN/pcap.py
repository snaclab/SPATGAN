#coding=utf-8
import csv
import dpkt
import socket
import sys
from datetime import datetime


def pcap_read(filename):

    cnt = 0
    ipcnt = 0
    tcpcnt = 0
    udpcnt = 0
    server_ip = '192.168.1.133'
    client_ip = '124.123.243.15'
    new_name = filename[:-5]
    print(new_name)

    with open(new_name + '.txt', 'w') as text_file:
        for ts, pkt in dpkt.pcap.Reader(open(filename, 'rb')):
            cnt += 1
            eth = dpkt.ethernet.Ethernet(pkt) 
            if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                continue
            ip = eth.data
            
            # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
            do_not_fragment = bool(ip.off & dpkt.ip.IP_DF)
            more_fragments = bool(ip.off & dpkt.ip.IP_MF)
            fragment_offset = ip.off & dpkt.ip.IP_OFFMASK

            # Print out the info
            if ip.p == dpkt.ip.IP_PROTO_UDP or ip.p == dpkt.ip.IP_PROTO_TCP:
                if (socket.inet_ntoa(ip.src) == server_ip and socket.inet_ntoa(ip.dst) == client_ip) or \
                (socket.inet_ntoa(ip.src) == client_ip and socket.inet_ntoa(ip.dst) == server_ip):
                    print('Packet#%s ' % str(cnt).ljust(5), end='')
                    print('IP: %s -> %s ' % (socket.inet_ntoa(ip.src).ljust(15), socket.inet_ntoa(ip.dst).ljust(15)), end='')
                    print('Size: %s ' % str(len(pkt)).ljust(5), end='')
                    print('Time: %s ' % str(datetime.utcfromtimestamp(ts)), end='')
                    print('Protocol: %s ' % ip.p, end='')
                    print('Data Size: %s ' % str(len(ip.data.data)).ljust(4), end='')
                    print('Header Size: %s' % str(len(pkt) - len(ip.data.data)))
                    # print('Len: %s TTL: %s DF: %d MF: %d OFFSET: %d' % (str(ip.len).ljust(5), str(ip.ttl).ljust(3), do_not_fragment, more_fragments, fragment_offset))
                    text_file.write('%s %s %s %s %s %s\n' % \
                        (str(cnt), socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst), str(len(pkt)), str(len(pkt) - len(ip.data.data)),
                            str(datetime.utcfromtimestamp(ts))))

            ipcnt += 1
            if ip.p == dpkt.ip.IP_PROTO_TCP: 
                tcpcnt += 1

            if ip.p == dpkt.ip.IP_PROTO_UDP:
                udpcnt += 1

    with open(new_name + '.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(' ') for line in stripped if line)
        
        with open(new_name + '.csv', 'w', newline='') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('number', 'src', 'dst', 'size', 'clean_size', 'date', 'time'))
            writer.writerows(lines)

    print('Total number of packets in the pcap file: ', cnt)
    print('Total number of ip packets: ', ipcnt)
    print('Total number of tcp packets: ', tcpcnt)
    print('Total number of udp packets: ', udpcnt)

if __name__ == '__main__':
    # get model
    if len(sys.argv) > 2:
        print('python3 pcap.py <pcap file>')
    else:
        pcap_read(sys.argv[1])

# Ref: https://stackoverflow.com/questions/18256342/parsing-a-pcap-file-in-python