import taichi as ti


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


def auto_cast(base_obj):
    from base import Base
    if not isinstance(base_obj, Base):
        print("TypeError: Not Subclass of 'Base', exiting..")
        exit(0)
    if type(base_obj) != Base:
        return base_obj
    from area import NeuronArea
    from sense import NeuronSense
    from connection import NeuronConnection
    from action import NeuronAction
    '''
    TODO: add future class for auto-casting
    '''
    class_name = base_obj.class_name
    obj = eval(class_name+'()')
    for _item_name in dir(base_obj):
        if _item_name[0] != '_':
            setattr(obj, _item_name, getattr(base_obj, _item_name))
    return obj


def recursive_cast(obj):
    if hasattr(obj, "children") and isinstance(obj.children, list):
        for _ in len(obj.children):
            new_child = auto_cast(obj.children.pop(0))
            obj.children.append(new_child)
            recursive_cast(obj)
