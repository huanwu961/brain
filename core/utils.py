def json_check(obj):
    success = True
    if isinstance(obj, dict):
        for key, val in obj.items():
            success = success and isinstance(val, (str, int, float, bool))
    elif isinstance(obj, list):
        for item in obj:
            success = success and isinstance(item, (str, int, float, bool))
    elif isinstance(obj, (str, int, float, bool)):
        success = True
    else:
        success = False
    return success