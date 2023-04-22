import os
import ipaddress
import urllib.request


def get_path_to_file_from_root(path):
    path_to_file = os.path.normpath(
        os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))) + "\\" + path
    )
    return path_to_file

def get_location_by_ip(addr=''):
    from urllib.request import urlopen
    from json import load
    if addr == '':
        url = 'https://ipinfo.io/json'
    else:
        url = 'https://ipinfo.io/' + addr + '/json'
    res = urlopen(url)
    #response from url(if res==None then check connection)
    data = load(res)
    return data["loc"].split(',')


def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0]
    else:
        ip = request.META.get("REMOTE_ADDR")
    ip_service = ipaddress.IPv4Address(ip)
    if ip_service.is_link_local or ip_service.is_private:
        ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
    return ip
