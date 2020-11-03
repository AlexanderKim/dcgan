import torch
from networks.Generator import Generator


class GeneratorModel(object):
    def __init__(self, gen_weights_path):
        self.gen = Generator()
        self.gen.load_state_dict(torch.load(gen_weights_path))

    def generate(self, n_samples):
        return self.gen(self.gen.gen_noize(n_samples))
