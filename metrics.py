from torch.nn import functional as F
import torch

def JSDiv(p, q):
    m = (p+q)/2
    return 0.5*F.kl_div(torch.log(p), m) + 0.5*F.kl_div(torch.log(q), m)

# ZRF/UnLearningScore
def UnLearningScore(tmodel, gold_model, forget_dl, batch_size, device):
    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for batch in forget_dl:
            x, y, cy = batch
            x = x.to(device)
            model_output = tmodel(x)
            gold_model_output = gold_model(x)
            model_preds.append(F.softmax(model_output, dim = 1).detach().cpu())
            gold_model_preds.append(F.softmax(gold_model_output, dim = 1).detach().cpu())
    
    
    model_preds = torch.cat(model_preds, axis = 0)
    gold_model_preds = torch.cat(gold_model_preds, axis = 0)
    return 1-JSDiv(model_preds, gold_model_preds)
