import torch

class Trainer():

    def __init__(self, device, model, loss_fn, optimizer, config):
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # 예측 오류 계산
            pred = model(X)
            loss = loss_fn(pred, y)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def trainer(self, train_dataloader, test_dataloader):
        for t in range(self.config.n_epochs):
            self.train(train_dataloader, self.model, self.loss_fn, self.optimizer)
            self.test(test_dataloader, self.model, self.loss_fn)

        torch.save(self.model.state_dict(), "model.pth")