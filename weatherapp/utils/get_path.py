import os


def get_path_to_file_from_root(path):
    path_to_file = os.path.normpath(
        os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))) + "\\" + path
    )
    return path_to_file


def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0]
    else:
        ip = request.META.get("REMOTE_ADDR")
    return ip
