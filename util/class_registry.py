import importlib


class ClassRegistry(type):
    module_path = None

    def __init__(cls, name, bases, nmspc):
        super(ClassRegistry, cls).__init__(name, bases, nmspc)
        if not hasattr(cls, 'registry') and object in bases:
            cls.registry = {}
            return
        if hasattr(cls, 'name'):
            cls.registry[cls.name] = cls

    def create(cls, name, *args, **kwargs):
        m = importlib.import_module(cls.module_path)
        for submodule_name in m.__all__:
            importlib.import_module('{}.{}'.format(cls.module_path, submodule_name))
        if name not in cls.registry:
            raise ModuleNotFoundError(
                'No class named "{}" registered'.format(name))
        return cls.registry[name](*args, **kwargs)
