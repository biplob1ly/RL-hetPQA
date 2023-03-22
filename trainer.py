
class Trainer:
    def __init__(self,
                 args,
                 generator,
                 retriever,
                 gen_optimizer,
                 ret_optimizer):
        self.args = args
        self.generator = generator
        self.retriever = retriever
        self.gen_optimizer = gen_optimizer
        self.ret_optimizer = ret_optimizer

    def train_generator(self, data_loader):
        self.generator.train()
        # for batch in data_loader:
