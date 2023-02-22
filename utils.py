import torch
from torch import nn
from torch.nn import functional as F


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))*100

def training_step(model, batch, device):
    images, labels, clabels = batch 
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)                  # Generate predictions
    loss = F.cross_entropy(out, clabels) # Calculate loss
    return loss

def validation_step(model, batch, device):
    images, labels, clabels = batch 
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)                    # Generate predictions
    loss = F.cross_entropy(out, clabels)   # Calculate loss
    acc = accuracy(out, clabels)           # Calculate accuracy
    return {'Loss': loss.detach(), 'Acc': acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}

def epoch_end(model, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['Loss'], result['Acc']))



@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs,  model, train_loader, val_loader, device, pretrained_lr=0.001, finetune_lr=0.01):
    torch.cuda.empty_cache()
    history = []
    
    try:
        param_groups = [
            {'params':model.base.parameters(),'lr':pretrained_lr},
            {'params':model.final.parameters(),'lr':finetune_lr}
        ]
        optimizer = torch.optim.Adam(param_groups)
    except:
        optimizer = torch.optim.Adam(model.parameters(), finetune_lr)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
        
        # Validation phase
        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        sched.step(result['Loss'])
    return history