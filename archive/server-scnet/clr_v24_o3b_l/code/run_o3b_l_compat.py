import pickle, types, runpy, torch
class NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)
pm = types.ModuleType("pickle_compat")
pm.Unpickler = NumpyCoreCompatUnpickler
pm.Pickler = pickle.Pickler
pm.dump = pickle.dump
pm.dumps = pickle.dumps
pm.load = pickle.load
pm.loads = pickle.loads
_original_torch_load = torch.load
def compatible_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is False and "pickle_module" not in kwargs:
        kwargs["pickle_module"] = pm
    return _original_torch_load(*args, **kwargs)
torch.load = compatible_torch_load
with open("/root/private_data/clr_v24_o3b_l/output/runtime_compatibility.txt", "w", encoding="utf-8") as f:
    f.write("Checkpoint NumPy 2 global-path compatibility: numpy._core -> numpy.core during torch.load only; checkpoint bytes unchanged.\n")
runpy.run_module("a1_v24_o3b_l", run_name="__main__")
