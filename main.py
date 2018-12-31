import torchvision.transforms as transforms

from pyro.infer import SVI, Trace_ELBO
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pyro.set_rng_seed(0)

# dataset
train_set = torch.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_set = torch.datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 10, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = 10, shuffle = True)


model = VAE()
optim = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(vae.model, vae.guide, optim, loss = Trace_ELBO())

# train
print("Starting training")
