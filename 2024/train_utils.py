import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def train(dataloader, model, loss_fn, optimizer):
    """
    Mini-batched training
    
    Args:
        dataloader ()     : 
        model (nn.Module) : 
        loss_fn ()        : loss/objective function
        optimizer ()      : optimization algorithm
    """
    size = len(dataloader.dataset)
    model.train()
    
    model = torch.compile(model)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}]/{size:>5d}")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    model = torch.compile(model)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    
    return 100*correct, test_loss
    # print(f"[Test] \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
def train_uns(dataloader, model, loss_fn, optimizer):
    """
    Mini-batched training
    
    Args:
        dataloader ()     : 
        model (nn.Module) : 
        loss_fn ()        : loss/objective function
        optimizer ()      : optimization algorithm
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        X = X.to(DEVICE)
        # Compute prediction error
        
        (n, c, d1, d2) = X.shape
        X = torch.reshape(X, (-1, c*d1*d2))
        # print("tu train_uns, X shape:", X.shape)
        Xrec = model(X)
        # print("tu train_uns, Xrec shape:", Xrec.shape)
        loss = loss_fn(Xrec, X)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}]/{size:>5d}")
            
def test_uns(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(DEVICE)
            
            (n, c, d1, d2) = X.shape
            X = torch.reshape(X, (-1, c*d1*d2))
            Xrec = model(X)
            
            test_loss += loss_fn(Xrec, X).item()
    
    test_loss /= num_batches
    
    return test_loss