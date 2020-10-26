import torch


class Generator_model(object):
    def __init__(self, gen_weights):
        from networks.Generator import Generator
        self.gen = Generator()
        self.gen.load_state_dict(gen_weights)

    def generate(self, n_samples):
        return self.gen(self.gen.gen_noize(n_samples))