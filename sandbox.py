import torch
import matplotlib.pyplot as plt
from growing import MLP
import numpy as np
from contextlib import contextmanager

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


class SimpleModel(torch.nn.Module):
    def __init__(self, n_in, n_out, h, a, b):
        super().__init__()
        self.a = MLP(n_in, h, a)
        self.b = MLP(h, n_out, b)
        self.dropout = torch.nn.Dropout()

    def forward(self, x):
        x = self.a(x)
        x = self.dropout(x)
        x = self.b(x)
        return x

    def grow(self ,*args, **kw):
        return self.a.grow(*args, **kw), self.b.grow(*args, **kw)

    def select(self, *args, **kw):
        return self.a.select(*args, **kw), self.b.select(*args, **kw)

    def degrow(self, sa, sb):
        return self.a.degrow(sa), self.b.degrow(sb)
    
    @contextmanager
    def new_grad_only(self):
        with self.a.new_grad_only(), self.b.new_grad_only():
            yield

def toy_data(n=16):
    # toy training data
    train_x = torch.Tensor(n, 1)
    torch.nn.init.uniform_(train_x, -5, 5)
    train_y = torch.Tensor(n, 1)
    torch.nn.init.normal_(train_y, 0, .1)
    train_y += torch.sin(train_x)
    return train_x, train_y

def growing_train(model, *, grow=None, lr=0.01, use_onecycle=False, num_batches=200, num_epochs=10):

    losses = list()
    criterion = torch.nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    for growth_phase in range(5):  
        if growth_phase > 0:
            if grow is not None:
                grow(model)
                optim = torch.optim.Adam(model.parameters(), lr=lr)        

        scheduler = None
                    
        if use_onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=200, max_lr=lr)
        
        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in range(num_batches):
                train_x, train_y = toy_data()

                optim.zero_grad()

                y = model(train_x)
                loss = criterion(y, train_y)

                loss.backward()
                optim.step()

                if scheduler:
                    scheduler.step()
                
                loss_sum += loss.data

            losses.append(loss_sum / num_batches)
    return dict(
        model=model,
        loss_series=np.array(losses),
        size=sum(p.numel() for p in model.parameters()),
    )

def experiment_series(model_args, runs=5, **kw):
    results = list()
    plt.subplots(figsize=(20, 8))
    
    for i in range(runs):
        torch.manual_seed(i)
        model = SimpleModel(*model_args)
        result = growing_train(model, **kw)

        print(f"#{i}: loss: {result['loss_series'][-1]} - size: {result['size']}")

        result['i'] = i
        results.append(result)
        
        
        plt.plot(result['loss_series'])
        plt.gca().xaxis.grid(True)
    plt.show()
    return results
        
    
def eval_model(model):
    train_x, train_y = toy_data()
    
    plt.subplots(figsize=(12, 8))

    x = torch.linspace(-5, 5, 100)[:, None]

    model.eval()

    y = model(x).detach().numpy()
    x = x.detach().numpy()
    plt.plot(x, y, c='tab:orange')

    plt.scatter(train_x.detach(), train_y.detach(), marker='+')
    #plt.legend()
    plt.show()
    
def eval_series(results):
    train_x, train_y = toy_data()
    
    plt.subplots(figsize=(12, 8))

    x = torch.linspace(-5, 5, 100)[:, None]

    for r in results:
        model = r['model']
    
        model.eval()

        y = model(x).detach().numpy()
        plt.plot(x.detach().numpy(), y)

    plt.scatter(train_x.detach(), train_y.detach(), marker='+', c='k')
    #plt.legend()
    plt.show()
    
    
