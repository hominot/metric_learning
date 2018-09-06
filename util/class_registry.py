class ClassRegistry(type):
    def __init__(cls, name, bases, nmspc):
        super(ClassRegistry, cls).__init__(name, bases, nmspc)
        if not hasattr(cls, 'registry'):
            cls.registry = dict()
        if hasattr(cls, 'name'):
            if cls.name in cls.registry:
                pass
                """
                raise Exception(
                    '"{}" already registered as "{}"'.format(
                        cls.name, cls.registry[cls.name].__name__))
                """

            cls.registry[cls.name] = cls

    def __iter__(cls):
        return iter(cls.registry.values())

    def __str__(cls):
        if cls in cls.registry.values():
            return cls.name
        return cls.__name__ + ': ' + ', '.join([sc.name for sc in cls])

    def create(cls, name, *args, **kwargs):
        if name not in cls.registry:
            raise ModuleNotFoundError(
                'No {} named "{}" registered'.format(cls.__name__, name))
        return cls.registry[name](*args, **kwargs)
