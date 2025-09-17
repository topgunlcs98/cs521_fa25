import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 0 # target class

epsReal = 0.5  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
loss.backward()

# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad

def targeted_ifgsm(model, x, target, eps=0.5, alpha=0.02, iters=40, clamp_min=0.0, clamp_max=1.0):
    x_orig = x.clone().detach()
    adv = x_orig.clone().detach()
    adv.requires_grad_()
    loss_fn = nn.CrossEntropyLoss()

    for i in range(iters):
        logits = model(adv)
        loss = loss_fn(logits, torch.tensor([target], dtype=torch.long))
        model.zero_grad()
        if adv.grad is not None:
            adv.grad.zero_()
        loss.backward()
        with torch.no_grad():
            adv = adv - alpha * adv.grad.sign()
            adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps)
            adv = torch.clamp(adv, clamp_min, clamp_max)
            adv.requires_grad_()
    return adv.detach()

adv_x = targeted_ifgsm(N, x, target=1, eps=0.5, alpha=0.50, iters=200, clamp_min=0.0, clamp_max=1.0)

new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)

