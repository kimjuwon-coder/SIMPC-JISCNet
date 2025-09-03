import pkgutil

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    module_spec = loader.find_spec(module_name)
    module = module_spec.loader.load_module(module_name)
    exec('%s = module' % module_name)
