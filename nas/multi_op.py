class MultipleOptimizer(object):
    def __init__(self,*op):
        self.optimizer=op
    def zero_grad(self):
        for op in self.optimizer:
            op.zero_grad()

    def step(self):
        for op in self.optimizer:
            op.step()