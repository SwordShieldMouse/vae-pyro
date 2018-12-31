import torchvision.transforms as transforms

from pyro.infer import SVI, Trace_ELBO
from architectures import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pyro.set_rng_seed(0)

# dataset
train_set = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 10, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = 10, shuffle = True)


model = VAE().to(device)
optim = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(model.model, model.guide, optim, loss = Trace_ELBO())
epochs = 2

# train
print("starting training")
for epoch in range(epochs):
    epoch_loss = 0
    print("starting epoch {}".format(epoch))
    for ix, (x, _) in enumerate(train_loader):
        if ix % 500 == 0:
            print("on sample {}".format(ix))
        epoch_loss += svi.step(x)
    print("loss for epoch {} is {}".format(epoch, epoch_loss / len(train_loader.dataset)))

print("starting evaluation")
test_loss = 0
for ix, (x, _) in enumerate(test_loader):
    test_loss += svi.evaluate_loss(x)
print("test loss is {}".format(test_loss / len(test_loader.dataset)))
