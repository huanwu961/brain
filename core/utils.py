def json_check(obj):
    success = True
    if isinstance(obj, dict):
        for key, val in obj:
            success = success and json_check(val)
    if isinstance(obj, list):
        for item in obj:
            success = success and json_check(item)
    if isinstance(obj, (str, int, float, bool)):
        success = True
    else:
        success = False
    return success


def base_check(base, tar_type):
    success = True
    for child in base.children:
        success = success and base_check(child, tar_type)
    if isinstance(base, tar_type):
        success = True
    else:
        success = False
    return success
