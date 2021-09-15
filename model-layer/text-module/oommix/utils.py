import numpy as np


class Collector:
    def __init__(self):
        self.activations = {}
        self.handles = {}

    def create_hook_fn(self, name, fn):
        def hook(model, input, output):
            self.activations[name] = fn(output)

        return hook

    def collect_representation(self, model):
        target_names = ["embedding_norm"]
        target_names += ["encoder.%d.ff_norm" % i for i in range(12)]
        for name, module in model.named_modules():
            if name in target_names:
                hook_fn = self.create_hook_fn(name, lambda x: x.detach().cpu())
                self.handles[name] = module.register_forward_hook(hook_fn)

    def collect_attention(self, model):
        target_names = ["encoder.%d.mhsa" % i for i in range(12)]
        for name, module in model.named_modules():
            if name in target_names:
                hook_fn = self.create_hook_fn(
                    name + "_attn", lambda x: x[1].detach().cpu()
                )
                self.handles[name] = module.register_forward_hook(hook_fn)

    def remove_all_hook(self):
        for handle in self.handles.values():
            handle.remove()


def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q
