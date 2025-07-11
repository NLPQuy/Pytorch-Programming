# PyTorch Fundamentals - 100 Multiple Choice Questions

This comprehensive test includes 100 multiple choice questions about PyTorch fundamentals, arranged from **Easy → Intermediate → Hard → Very Hard (Mathematical Reasoning)**.

**Instructions:**
- Choose the best answer for each question
- Each question has 4 choices (A, B, C, D)
- Answers are provided at the end of the test

---

## **PART I: EASY QUESTIONS (Questions 1-15)**

**Question 1:** To create a 3x4 tensor filled with zeros, which command should we use?
A) `torch.empty(3, 4)`
B) `torch.zeros(3, 4)`
C) `torch.ones(3, 4)`
D) `torch.randn(3, 4)`

**Question 2:** Which function is used to create a tensor from a Python list?
A) `torch.from_list()`
B) `torch.tensor()`
C) `torch.create()`
D) `torch.make_tensor()`

**Question 3:** To get the shape of a tensor `x`, we use:
A) `x.size()`
B) `x.length()`
C) `x.dimensions()`
D) `x.get_shape()`

**Question 4:** Which tensor has shape (2, 3)?
A) `torch.zeros(3, 2)`
B) `torch.ones(2, 3)`
C) `torch.rand(6)`
D) `torch.randn(2, 3, 1)`

**Question 5:** To move a tensor to GPU, we use:
A) `tensor.gpu()`
B) `tensor.cuda()`
C) `tensor.to_gpu()`
D) `tensor.device('cuda')`

**Question 6:** Which operation performs element-wise multiplication between two tensors?
A) `torch.matmul(a, b)`
B) `torch.dot(a, b)`
C) `a * b`
D) `torch.cross(a, b)`

**Question 7:** To create a tensor with values from 0 to 9, we use:
A) `torch.range(0, 9)`
B) `torch.arange(10)`
C) `torch.linspace(0, 9, 10)`
D) `torch.sequence(10)`

**Question 8:** The function `torch.randn()` creates tensors with which distribution?
A) Uniform [0, 1]
B) Normal (0, 1)
C) Exponential
D) Bernoulli

**Question 9:** To reshape tensor `x` with 12 elements into shape (3, 4), we use:
A) `x.reshape(3, 4)`
B) `x.resize(3, 4)`
C) `x.change_shape(3, 4)`
D) `x.reform(3, 4)`

**Question 10:** Which method moves a tensor from GPU to CPU?
A) `tensor.cpu()`
B) `tensor.to_cpu()`
C) `tensor.device('cpu')`
D) `tensor.move_cpu()`

**Question 11:** To access the element at position (1, 2) of a 2D tensor `x`, we write:
A) `x(1, 2)`
B) `x[1][2]`
C) `x[1, 2]`
D) `x.get(1, 2)`

**Question 12:** Which function computes the sum of all elements in a tensor?
A) `torch.total()`
B) `torch.sum()`
C) `torch.add_all()`
D) `torch.accumulate()`

**Question 13:** To create a 4x4 identity matrix tensor, we use:
A) `torch.identity(4)`
B) `torch.eye(4)`
C) `torch.unit(4)`
D) `torch.diag(4)`

**Question 14:** Gradients are automatically computed when a tensor has the attribute:
A) `requires_grad=True`
B) `auto_grad=True`
C) `grad_enabled=True`
D) `compute_grad=True`

**Question 15:** To clone a tensor `x`, we use:
A) `x.copy()`
B) `x.clone()`
C) `x.duplicate()`
D) `torch.copy(x)`

---

## **PART II: INTERMEDIATE QUESTIONS (Questions 16-35)**

**Question 16:** Broadcasting allows operations between tensor with shape (3, 1) and tensor with which shape?
A) (2, 3)
B) (3, 4)
C) (4, 3)
D) (1, 4)

**Question 17:** To transpose a 2D tensor `x`, we use:
A) `x.transpose()`
B) `x.T`
C) `x.transpose(0, 1)`
D) Both B and C are correct

**Question 18:** The difference between `view()` and `reshape()` methods is:
A) `view()` always creates a new copy
B) `reshape()` only works with contiguous tensors
C) `view()` requires the tensor to be contiguous
D) There is no difference

**Question 19:** To concatenate two tensors `a` and `b` along dimension 0, we use:
A) `torch.cat([a, b], dim=0)`
B) `torch.concat([a, b], axis=0)`
C) `torch.join([a, b], dim=0)`
D) `torch.merge([a, b], dim=0)`

**Question 20:** The function `torch.squeeze()` does:
A) Add dimensions of size 1
B) Remove dimensions of size 1
C) Compress tensor data
D) Reduce tensor precision

**Question 21:** In PyTorch, `nn.Module` is:
A) A function to create modules
B) Base class for all neural network modules
C) Package containing neural networks
D) Decorator for functions

**Question 22:** To define a linear layer with 10 input features and 5 output features:
A) `nn.Linear(5, 10)`
B) `nn.Linear(10, 5)`
C) `nn.Dense(10, 5)`
D) `nn.FullyConnected(10, 5)`

**Question 23:** The ReLU activation function is implemented as:
A) `max(0, x)`
B) `min(0, x)`
C) `1 / (1 + exp(-x))`
D) `tanh(x)`

**Question 24:** To compute gradients of a loss function, we call:
A) `loss.gradient()`
B) `loss.backward()`
C) `loss.compute_grad()`
D) `torch.grad(loss)`

**Question 25:** An SGD optimizer with learning rate 0.01 is created using:
A) `torch.optim.SGD(model.parameters(), lr=0.01)`
B) `torch.optim.SGD(model, learning_rate=0.01)`
C) `torch.SGD(model.parameters(), lr=0.01)`
D) `torch.optimizer.SGD(model, lr=0.01)`

**Question 26:** To reset gradients to zero before each backward pass:
A) `model.zero_grad()`
B) `optimizer.zero_grad()`
C) `torch.zero_grad()`
D) `gradient.reset()`

**Question 27:** The `torch.no_grad()` context manager is used to:
A) Catch gradient errors
B) Disable gradient computation
C) Clear all gradients
D) Debug gradient flow

**Question 28:** 2D Convolution with kernel size 3x3, stride 1, padding 1 applied to 32x32 input gives output size:
A) 30x30
B) 32x32
C) 34x34
D) 16x16

**Question 29:** MaxPool2d with kernel size 2x2 reduces spatial dimensions by:
A) 1/2
B) 1/4
C) 1/8
D) No change

**Question 30:** To save model state, we use:
A) `torch.save(model, 'model.pth')`
B) `torch.save(model.state_dict(), 'model.pth')`
C) `model.save('model.pth')`
D) `torch.export(model, 'model.pth')`

**Question 31:** DataLoader in PyTorch is used to:
A) Load data from files
B) Create batches and shuffle data
C) Preprocess data
D) Validate data

**Question 32:** To switch model to evaluation mode:
A) `model.eval()`
B) `model.evaluation()`
C) `model.test_mode()`
D) `model.inference()`

**Question 33:** CrossEntropyLoss in PyTorch combines:
A) Softmax + MSE Loss
B) Sigmoid + Binary Cross Entropy
C) LogSoftmax + NLL Loss
D) ReLU + L1 Loss

**Question 34:** To create a tensor on GPU from the beginning:
A) `torch.zeros(3, 4).cuda()`
B) `torch.zeros(3, 4, device='cuda')`
C) `torch.cuda.zeros(3, 4)`
D) Both A and B are correct

**Question 35:** Batch normalization is usually applied:
A) Before activation function
B) After activation function
C) Instead of activation function
D) Position doesn't matter

---

## **PART III: HARD QUESTIONS (Questions 36-50)**

**Question 36:** In advanced indexing, `tensor[torch.arange(3), [0, 2, 1]]` will:
A) Get elements at (0,0), (1,2), (2,1)
B) Get elements at (0,1), (2,1), (1,0)
C) Produce an error
D) Get all elements

**Question 37:** Memory layout of tensors affects:
A) Only performance
B) Only correctness of operations
C) Both performance and some operations
D) Nothing

**Question 38:** To create a custom dataset in PyTorch, we need to implement:
A) `__init__` and `__len__`
B) `__init__` and `__getitem__`
C) `__init__`, `__len__` and `__getitem__`
D) Only `__getitem__`

**Question 39:** Gradient clipping is used to:
A) Speed up training
B) Reduce memory usage
C) Prevent exploding gradients
D) Increase accuracy

**Question 40:** `torch.autograd.Function` is used to:
A) Automatically compute gradients
B) Define custom gradient computation
C) Debug gradient flow
D) Optimize gradient computation

**Question 41:** Attention mechanism in transformers computes:
A) `softmax(QK^T/√d)V`
B) `softmax(QK^T)V`
C) `QK^TV`
D) `softmax(Q)K^TV`

**Question 42:** Mixed precision training uses:
A) Only float16
B) Only float32
C) Combination of float16 and float32
D) Only int8

**Question 43:** Dynamic computation graph means:
A) Graph is created at compile time
B) Graph is created at runtime
C) Graph can change size
D) Graph is automatically optimized

**Question 44:** To implement residual connection, we need:
A) `output = layer(input)`
B) `output = layer(input) + input`
C) `output = layer(input) * input`
D) `output = concat(layer(input), input)`

**Question 45:** Distributed training with DDP uses:
A) Data parallelism
B) Model parallelism
C) Pipeline parallelism
D) Tensor parallelism

**Question 46:** `torch.jit.script()` is used to:
A) Debug code
B) Profile performance
C) Compile model to TorchScript
D) Serialize model

**Question 47:** The most common learning rate scheduling is:
A) Constant learning rate
B) Step decay
C) Cosine annealing
D) Exponential decay

**Question 48:** To implement custom loss function with gradient support:
A) Inherit from `nn.Module`
B) Inherit from `torch.autograd.Function`
C) Write regular Python function
D) Both A and B are possible

**Question 49:** Model ensembling in PyTorch is commonly implemented using:
A) Averaging predictions
B) Voting
C) Stacking
D) All of the above

**Question 50:** To optimize memory usage when training large models:
A) Gradient checkpointing
B) Mixed precision
C) Model sharding
D) All of the above

---

## **PART IV: VERY HARD QUESTIONS - ADVANCED PYTORCH COMPUTATIONS (Questions 51-100)**

**Question 51:** Advanced tensor indexing with conditions:
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = x > 5
result = x[mask]
```
What is the shape of `result`?
A) (3, 3)
B) (1, 3) 
C) (3,)
D) (4,)

**Question 52:** Efficient matrix multiplication with broadcasting:
```python
A = torch.randn(32, 128, 64)  # batch of matrices
B = torch.randn(64, 256)      # single matrix
result = torch.matmul(A, B)
```
What is the shape of `result`?
A) (32, 128, 256)
B) (32, 64, 256)
C) (128, 256)
D) Error - incompatible shapes

**Question 53:** Memory-efficient gradient computation:
```python
def efficient_forward(x, weight):
    with torch.no_grad():
        intermediate = torch.relu(x @ weight)
    intermediate.requires_grad_(True)
    output = torch.sum(intermediate ** 2)
    return output, intermediate
```
This technique is called:
A) Gradient checkpointing
B) Mixed precision
C) Gradient accumulation
D) Memory mapping

**Question 54:** Advanced Einstein summation:
```python
A = torch.randn(10, 3, 4)
B = torch.randn(4, 5)
result = torch.einsum('bij,jk->bik', A, B)
```
What is the shape of `result`?
A) (10, 3, 5)
B) (10, 4, 5)
C) (3, 5)
D) (10, 3, 4)

**Question 55:** Custom backward function implementation:
```python
class PowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, power):
        ctx.save_for_backward(input)
        ctx.power = power
        return input ** power
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        power = ctx.power
        return grad_output * power * (input ** (power - 1)), None
```
The second return value is `None` because:
A) Power doesn't need gradients
B) It's an integer parameter
C) ctx.power is not a tensor
D) All of the above

**Question 56:** Efficient batched operations:
```python
def batched_matrix_inverse(matrices):
    # matrices: (batch_size, n, n)
    return torch.linalg.inv(matrices)

# Alternative approach:
def manual_batched_inverse(matrices):
    batch_size = matrices.size(0)
    results = []
    for i in range(batch_size):
        results.append(torch.inverse(matrices[i]))
    return torch.stack(results)
```
Which approach is more efficient?
A) `batched_matrix_inverse` - vectorized operations
B) `manual_batched_inverse` - more control
C) Both are equivalent
D) Depends on batch size

**Question 57:** Advanced tensor reshaping and memory layout:
```python
x = torch.randn(2, 3, 4, 5)
y = x.permute(0, 2, 1, 3)
z = y.contiguous()
```
After these operations:
A) `y` and `z` have the same memory layout
B) `y` and `z` have different memory layouts
C) `z.is_contiguous()` returns `True`, `y.is_contiguous()` returns `False`
D) Both B and C are correct

**Question 58:** Efficient element-wise operations with different dtypes:
```python
a = torch.randn(1000, 1000, dtype=torch.float32)
b = torch.randn(1000, 1000, dtype=torch.float64)
result = a + b
```
What happens and what is `result.dtype`?
A) Error - incompatible dtypes
B) `result.dtype` is `torch.float32`
C) `result.dtype` is `torch.float64` 
D) `result.dtype` is automatically chosen

**Question 59:** Advanced gradient accumulation:
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```
Why divide loss by `accumulation_steps`?
A) To prevent gradient explosion
B) To maintain equivalent learning dynamics
C) To reduce memory usage
D) To speed up training

**Question 60:** Efficient tensor concatenation:
```python
tensors = [torch.randn(100, 50) for _ in range(10)]

# Method 1:
result1 = torch.cat(tensors, dim=0)

# Method 2:
result2 = torch.stack(tensors).view(-1, 50)
```
Which method is more memory efficient?
A) Method 1 (`torch.cat`)
B) Method 2 (`torch.stack` + `view`)
C) Both are equivalent
D) Depends on tensor sizes

**Question 61:** Advanced indexing with multiple conditions:
```python
x = torch.randn(100, 100)
row_mask = (torch.arange(100) % 2 == 0)
col_mask = (torch.arange(100) % 3 == 0)
result = x[row_mask][:, col_mask]
```
What is the shape of `result`?
A) (50, 34)
B) (50, 33)
C) (100, 33)
D) (50, 100)

**Question 62:** Custom CUDA kernel integration:
```python
import torch
from torch.utils.cpp_extension import load

# Load custom CUDA kernel
custom_op = load(name="custom_op", sources=["custom_kernel.cu"])

def custom_function(input_tensor):
    return custom_op.forward(input_tensor)
```
To make this support autograd:
A) Wrap in `torch.autograd.Function`
B) Use `torch.autograd.grad`
C) Define backward pass in CUDA
D) Both A and C

**Question 63:** Efficient sparse tensor operations:
```python
indices = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
values = torch.FloatTensor([3, 4, 5])
sparse_tensor = torch.sparse.FloatTensor(indices, values, (2, 3))

dense_result = sparse_tensor.to_dense()
```
What does `dense_result` look like?
A) `[[0, 0, 3], [4, 0, 5]]`
B) `[[3, 4, 5], [0, 0, 0]]`
C) `[[0, 4, 0], [0, 0, 5]]`
D) Error - invalid indices

**Question 64:** Advanced optimizer state manipulation:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train for some steps...

# Modify learning rate for specific parameter groups
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1

# Access momentum buffers
for group in optimizer.param_groups:
    for p in group['params']:
        state = optimizer.state[p]
        if 'exp_avg' in state:
            momentum = state['exp_avg']
```
This technique is useful for:
A) Learning rate scheduling
B) Gradient analysis
C) Transfer learning
D) All of the above

**Question 65:** Memory-efficient model checkpointing:
```python
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
```
Using `map_location='cpu'` is beneficial because:
A) Reduces memory usage on GPU
B) Enables loading on different devices
C) Prevents CUDA memory fragmentation
D) All of the above

**Question 66:** Advanced tensor slicing and striding:
```python
x = torch.arange(24).view(2, 3, 4)
result = x[:, ::2, 1::2]
```
What is the shape and content of `result`?
A) Shape: (2, 2, 2), contains elements with specific stride pattern
B) Shape: (2, 1, 2), contains every other element
C) Shape: (1, 2, 2), first dimension reduced
D) Error - invalid slicing

**Question 67:** Efficient implementation of custom layers:
```python
class EfficientLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # Which is more efficient for large batches?
        # Option A:
        return torch.addmm(self.bias.unsqueeze(0), x, self.weight.t())
        # Option B:
        return F.linear(x, self.weight, self.bias)
```
A) Option A - explicit operations
B) Option B - optimized function
C) Both are equivalent
D) Depends on tensor sizes

**Question 68:** Advanced device management:
```python
def multi_gpu_computation(tensors):
    device_count = torch.cuda.device_count()
    results = []
    
    for i, tensor in enumerate(tensors):
        device = f'cuda:{i % device_count}'
        tensor = tensor.to(device)
        result = expensive_computation(tensor)
        results.append(result.cpu())
    
    return torch.cat(results, dim=0)
```
This pattern implements:
A) Data parallelism
B) Model parallelism
C) Manual load balancing
D) Pipeline parallelism

**Question 69:** Efficient batch processing with variable lengths:
```python
def collate_variable_length(batch):
    # batch: list of tensors with different lengths
    lengths = [len(item) for item in batch]
    max_length = max(lengths)
    
    padded = torch.zeros(len(batch), max_length, batch[0].size(-1))
    for i, item in enumerate(batch):
        padded[i, :len(item)] = item
    
    return padded, torch.tensor(lengths)
```
This is commonly used in:
A) NLP sequence processing
B) Time series data
C) Variable-size image processing
D) All of the above

**Question 70:** Advanced autograd hooks:
```python
def register_gradient_hooks(model):
    def hook_fn(module, grad_input, grad_output):
        print(f'Gradient norm for {module.__class__.__name__}: {grad_output[0].norm().item()}')
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.register_backward_hook(hook_fn)
```
This technique is useful for:
A) Debugging gradient flow
B) Gradient clipping per layer
C) Monitoring training dynamics
D) All of the above

**Question 71:** Efficient tensor comparison and masking:
```python
a = torch.randn(1000, 1000)
b = torch.randn(1000, 1000)

# Method 1:
mask1 = torch.where(a > b, torch.ones_like(a), torch.zeros_like(a))

# Method 2:
mask2 = (a > b).float()
```
Which method is more efficient?
A) Method 1 - explicit tensor creation
B) Method 2 - direct boolean conversion
C) Both are equivalent
D) Depends on sparsity of condition

**Question 72:** Advanced learning rate scheduling:
```python
def custom_lr_schedule(optimizer, epoch, warmup_epochs=5, total_epochs=100):
    if epoch < warmup_epochs:
        lr = 0.001 * epoch / warmup_epochs
    else:
        lr = 0.001 * 0.5 ** ((epoch - warmup_epochs) // 30)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```
This implements:
A) Warmup + step decay
B) Cosine annealing
C) Exponential decay
D) Linear scheduling

**Question 73:** Memory-efficient matrix operations:
```python
def efficient_large_matmul(A, B, chunk_size=1000):
    # A: (m, k), B: (k, n) where m, n are very large
    m, k = A.shape
    k, n = B.shape
    result = torch.zeros(m, n)
    
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        result[i:end_i] = torch.matmul(A[i:end_i], B)
    
    return result
```
This technique prevents:
A) Memory overflow
B) Numerical instability
C) GPU timeout
D) All of the above

**Question 74:** Advanced tensor broadcasting:
```python
a = torch.randn(3, 1, 4, 1)
b = torch.randn(1, 5, 1, 6)
result = a * b
```
What is the shape of `result`?
A) (3, 5, 4, 6)
B) (3, 1, 4, 6)
C) (1, 5, 4, 1)
D) Error - incompatible shapes

**Question 75:** Efficient gradient computation control:
```python
def selective_gradient_computation(model, input_data):
    # Only compute gradients for specific layers
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    output = model(input_data)
    loss = criterion(output, targets)
    loss.backward()
```
This technique is used in:
A) Transfer learning
B) Feature extraction
C) Reducing computation
D) All of the above

**Question 76:** Advanced tensor manipulation with view vs reshape:
```python
x = torch.randn(2, 3, 4)
y = x.transpose(1, 2)

# Which operations will work?
a = y.view(2, 12)      # Operation A
b = y.reshape(2, 12)   # Operation B
```
A) Both A and B work
B) Only B works (reshape is more flexible)
C) Only A works (view is faster)
D) Neither works

**Question 77:** Efficient data loading and preprocessing:
```python
class OptimizedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # Precompute expensive operations
        self.preprocessed_data = self.preprocess(data)
    
    def __getitem__(self, idx):
        # Return preprocessed data directly
        return self.preprocessed_data[idx]
    
    def preprocess(self, data):
        # Convert to tensor once, normalize, etc.
        return torch.stack([self.transform(item) for item in data])
```
This optimization:
A) Reduces training time per epoch
B) Increases memory usage
C) Improves CPU utilization
D) All of the above

**Question 78:** Advanced mixed precision training:
```python
scaler = torch.cuda.amp.GradScaler()
model = model.cuda()

for batch in dataloader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        output = model(batch)
        loss = criterion(output, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
The scaler is needed because:
A) FP16 has limited dynamic range
B) Gradients might underflow to zero
C) It maintains numerical stability
D) All of the above

**Question 79:** Efficient tensor padding and cropping:
```python
def smart_pad_crop(tensor, target_size):
    # tensor: (C, H, W), target_size: (target_H, target_W)
    c, h, w = tensor.shape
    target_h, target_w = target_size
    
    # Calculate padding/cropping
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    crop_h = max(0, h - target_h)
    crop_w = max(0, w - target_w)
    
    # Apply operations
    if crop_h > 0 or crop_w > 0:
        tensor = tensor[:, crop_h//2:h-crop_h//2, crop_w//2:w-crop_w//2]
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
    
    return tensor
```
This function handles:
A) Center cropping and padding
B) Arbitrary size transformation
C) Maintaining aspect ratio
D) All of the above

**Question 80:** Advanced parameter sharing:
```python
class SharedLayer(nn.Module):
    def __init__(self, shared_linear):
        super().__init__()
        self.shared_linear = shared_linear
    
    def forward(self, x):
        return self.shared_linear(x)

# Usage:
shared_layer = nn.Linear(128, 64)
layer1 = SharedLayer(shared_layer)
layer2 = SharedLayer(shared_layer)
```
Parameter sharing reduces:
A) Model size
B) Training time
C) Overfitting risk
D) All of the above

**Question 81:** Efficient batch normalization implementation:
```python
def manual_batch_norm(x, running_mean, running_var, weight, bias, 
                     training=True, momentum=0.1, eps=1e-5):
    if training:
        batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
        batch_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        
        # Update running statistics
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean.squeeze()
        running_var = (1 - momentum) * running_var + momentum * batch_var.squeeze()
        
        # Normalize using batch statistics
        x_norm = (x - batch_mean) / torch.sqrt(batch_var + eps)
    else:
        # Normalize using running statistics
        x_norm = (x - running_mean.view(1, -1, 1, 1)) / torch.sqrt(running_var.view(1, -1, 1, 1) + eps)
    
    return weight.view(1, -1, 1, 1) * x_norm + bias.view(1, -1, 1, 1)
```
The key difference between training and inference is:
A) Training uses batch statistics, inference uses running statistics
B) Training updates running statistics, inference doesn't
C) Momentum is only applied during training
D) All of the above

**Question 82:** Advanced tensor serialization:
```python
def efficient_tensor_save_load(tensor, filename):
    # Save efficiently
    torch.save({
        'data': tensor.cpu(),
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': str(tensor.device)
    }, filename)
    
    # Load efficiently
    checkpoint = torch.load(filename, map_location='cpu')
    loaded_tensor = checkpoint['data'].to(checkpoint['device'])
    return loaded_tensor
```
This approach ensures:
A) Device compatibility
B) Memory efficiency
C) Type preservation
D) All of the above

**Question 83:** Efficient model ensembling:
```python
def ensemble_predict(models, input_data):
    predictions = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            pred = model(input_data)
            predictions.append(F.softmax(pred, dim=1))
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred
```
Using `F.softmax` before averaging:
A) Ensures probabilities sum to 1
B) Improves ensemble performance
C) Makes averaging meaningful
D) All of the above

**Question 84:** Advanced gradient analysis:
```python
def analyze_gradients(model):
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += param.numel()
            print(f'{name}: {param_norm:.4f}')
    
    total_norm = total_norm ** (1. / 2)
    avg_norm = total_norm / param_count
    
    return total_norm, avg_norm
```
This analysis helps identify:
A) Vanishing gradient problems
B) Exploding gradient problems
C) Layer-wise learning dynamics
D) All of the above

**Question 85:** Efficient tensor operations with einsum:
```python
# Matrix multiplication using einsum
A = torch.randn(100, 200)
B = torch.randn(200, 300)

# Standard matmul
result1 = torch.matmul(A, B)

# Using einsum
result2 = torch.einsum('ij,jk->ik', A, B)
```
Einsum is particularly useful for:
A) Complex tensor contractions
B) Readable code for multi-dimensional operations
C) Broadcasting operations
D) All of the above

**Question 86:** Advanced learning rate finder:
```python
def find_learning_rate(model, dataloader, optimizer, criterion):
    lrs = []
    losses = []
    lr = 1e-8
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
        
        # Exponentially increase lr
        lr *= 1.1
        
        if lr > 1 or loss.item() > 4 * min(losses):
            break
    
    return lrs, losses
```
This technique helps:
A) Find optimal learning rate
B) Avoid manual hyperparameter tuning
C) Understand model sensitivity
D) All of the above

**Question 87:** Memory-efficient attention computation:
```python
def efficient_attention(query, key, value, chunk_size=1000):
    # query, key, value: (batch, seq_len, dim)
    batch_size, seq_len, dim = query.shape
    
    # Compute attention in chunks to save memory
    attention_output = torch.zeros_like(query)
    
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        
        q_chunk = query[:, i:end_i]  # (batch, chunk_size, dim)
        scores = torch.matmul(q_chunk, key.transpose(-2, -1))  # (batch, chunk_size, seq_len)
        weights = F.softmax(scores, dim=-1)
        output_chunk = torch.matmul(weights, value)  # (batch, chunk_size, dim)
        
        attention_output[:, i:end_i] = output_chunk
    
    return attention_output
```
This chunking strategy:
A) Reduces memory usage
B) Maintains mathematical equivalence
C) Enables processing longer sequences
D) All of the above

**Question 88:** Advanced optimizer warm-up:
```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```
Warmup prevents:
A) Initial gradient explosion
B) Poor early convergence
C) Training instability
D) All of the above

**Question 89:** Efficient model profiling:
```python
def profile_model(model, input_tensor, device='cuda'):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warm up
    for _ in range(10):
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Profile
    start_time = time.time()
    for _ in range(100):
        output = model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    
    # Memory usage
    memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return avg_time, memory_used
```
`torch.cuda.synchronize()` is needed because:
A) CUDA operations are asynchronous
B) Ensures accurate timing measurements
C) Waits for GPU operations to complete
D) All of the above

**Question 90:** Advanced tensor interpolation:
```python
def interpolate_tensors(tensor1, tensor2, alpha):
    # Linear interpolation between two tensors
    return (1 - alpha) * tensor1 + alpha * tensor2

def spherical_interpolation(tensor1, tensor2, alpha):
    # Spherical interpolation (for normalized tensors)
    dot_product = torch.sum(tensor1 * tensor2)
    theta = torch.acos(torch.clamp(dot_product, -1, 1))
    
    if theta.abs() < 1e-6:
        return interpolate_tensors(tensor1, tensor2, alpha)
    
    sin_theta = torch.sin(theta)
    return (torch.sin((1 - alpha) * theta) * tensor1 + 
            torch.sin(alpha * theta) * tensor2) / sin_theta
```
Spherical interpolation is useful for:
A) Interpolating normalized vectors
B) Smooth transitions in latent space
C) Preserving magnitude relationships
D) All of the above

**Question 91:** Efficient distributed gradient reduction:
```python
def manual_gradient_averaging(model, world_size):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                # Sum gradients across all processes
                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                # Average by dividing by world size
                param.grad.data /= world_size
```
This implements:
A) Data parallelism gradient synchronization
B) Manual distributed training
C) AllReduce communication pattern
D) All of the above

**Question 92:** Advanced tensor convolution:
```python
def manual_conv2d(input_tensor, kernel, stride=1, padding=0):
    # input_tensor: (batch, in_channels, height, width)
    # kernel: (out_channels, in_channels, kernel_h, kernel_w)
    
    batch_size, in_channels, in_h, in_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1
    
    # Apply padding
    if padding > 0:
        input_tensor = F.pad(input_tensor, (padding, padding, padding, padding))
    
    output = torch.zeros(batch_size, out_channels, out_h, out_w)
    
    for b in range(batch_size):
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    
                    patch = input_tensor[b, :, h_start:h_start+kernel_h, w_start:w_start+kernel_w]
                    output[b, oc, oh, ow] = torch.sum(patch * kernel[oc])
    
    return output
```
This manual implementation helps understand:
A) Convolution mathematics
B) Memory access patterns
C) Computational complexity
D) All of the above

**Question 93:** Efficient model quantization:
```python
def simple_quantization(tensor, num_bits=8):
    # Quantize tensor to specified number of bits
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Calculate scale and zero point
    scale = (max_val - min_val) / (2**num_bits - 1)
    zero_point = -min_val / scale
    
    # Quantize
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, 0, 2**num_bits - 1)
    
    # Dequantize
    dequantized = (quantized - zero_point) * scale
    
    return quantized.byte(), scale, zero_point
```
Quantization reduces:
A) Model size
B) Inference time
C) Memory bandwidth requirements
D) All of the above

**Question 94:** Advanced loss function implementation:
```python
def focal_loss(predictions, targets, alpha=1, gamma=2):
    # predictions: (batch_size, num_classes)
    # targets: (batch_size,)
    
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.mean()

def label_smoothing_loss(predictions, targets, smoothing=0.1):
    num_classes = predictions.size(1)
    confidence = 1.0 - smoothing
    
    # Create smooth labels
    smooth_targets = torch.zeros_like(predictions)
    smooth_targets.fill_(smoothing / (num_classes - 1))
    smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)
    
    return F.kl_div(F.log_softmax(predictions, dim=1), smooth_targets, reduction='batchmean')
```
These advanced loss functions:
A) Handle class imbalance (focal loss)
B) Prevent overconfidence (label smoothing)
C) Improve generalization
D) All of the above

**Question 95:** Efficient tensor comparison operations:
```python
def efficient_tensor_comparison(tensor1, tensor2, threshold=1e-6):
    # Multiple ways to compare tensors
    
    # Method 1: Element-wise absolute difference
    diff = torch.abs(tensor1 - tensor2)
    close_elementwise = diff < threshold
    
    # Method 2: Using torch.allclose
    close_allclose = torch.allclose(tensor1, tensor2, atol=threshold)
    
    # Method 3: Relative and absolute tolerance
    close_isclose = torch.isclose(tensor1, tensor2, atol=threshold, rtol=1e-5)
    
    return close_elementwise, close_allclose, close_isclose
```
`torch.allclose` vs `torch.isclose`:
A) `allclose` returns single boolean, `isclose` returns tensor
B) `allclose` is faster for full tensor comparison
C) `isclose` provides element-wise comparison details
D) All of the above

**Question 96:** Advanced model surgery:
```python
def transfer_weights(source_model, target_model, layer_mapping):
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    
    # Transfer matching layers
    for source_name, target_name in layer_mapping.items():
        if source_name in source_dict and target_name in target_dict:
            source_param = source_dict[source_name]
            target_param = target_dict[target_name]
            
            # Handle size mismatches
            if source_param.shape == target_param.shape:
                target_dict[target_name] = source_param.clone()
            else:
                print(f"Size mismatch: {source_name} {source_param.shape} -> {target_name} {target_param.shape}")
                # Partial transfer for compatible dimensions
                min_dims = [min(s, t) for s, t in zip(source_param.shape, target_param.shape)]
                target_dict[target_name][:min_dims[0], :min_dims[1]] = source_param[:min_dims[0], :min_dims[1]]
    
    target_model.load_state_dict(target_dict)
```
This technique enables:
A) Transfer learning between different architectures
B) Model adaptation
C) Partial weight initialization
D) All of the above

**Question 97:** Efficient attention mask computation:
```python
def create_attention_masks(sequence_lengths, max_length, device='cuda'):
    batch_size = len(sequence_lengths)
    
    # Create padding mask
    padding_mask = torch.zeros(batch_size, max_length, device=device)
    for i, length in enumerate(sequence_lengths):
        padding_mask[i, length:] = 1
    
    # Create causal mask (for autoregressive models)
    causal_mask = torch.triu(torch.ones(max_length, max_length, device=device), diagonal=1)
    
    # Combine masks
    combined_mask = padding_mask.unsqueeze(1) + causal_mask.unsqueeze(0)
    
    return combined_mask > 0
```
Attention masks are essential for:
A) Ignoring padding tokens
B) Preventing future information leakage
C) Efficient batch processing
D) All of the above

**Question 98:** Advanced tensor statistics:
```python
def compute_tensor_statistics(tensor, dim=None, keepdim=False):
    stats = {}
    
    # Basic statistics
    stats['mean'] = torch.mean(tensor, dim=dim, keepdim=keepdim)
    stats['std'] = torch.std(tensor, dim=dim, keepdim=keepdim)
    stats['var'] = torch.var(tensor, dim=dim, keepdim=keepdim)
    
    # Advanced statistics
    stats['median'] = torch.median(tensor, dim=dim, keepdim=keepdim)[0] if dim is not None else torch.median(tensor)
    stats['min'] = torch.min(tensor, dim=dim, keepdim=keepdim)[0] if dim is not None else torch.min(tensor)
    stats['max'] = torch.max(tensor, dim=dim, keepdim=keepdim)[0] if dim is not None else torch.max(tensor)
    
    # Percentiles
    if dim is None:
        flat_tensor = tensor.flatten()
        stats['q25'] = torch.quantile(flat_tensor, 0.25)
        stats['q75'] = torch.quantile(flat_tensor, 0.75)
    
    return stats
```
These statistics help with:
A) Understanding data distribution
B) Debugging numerical issues
C) Monitoring training dynamics
D) All of the above

**Question 99:** Efficient model compilation and optimization:
```python
def optimize_model_for_inference(model, example_input):
    # Set to evaluation mode
    model.eval()
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize the traced model
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    # Freeze the model
    frozen_model = torch.jit.freeze(optimized_model)
    
    return frozen_model

def compare_inference_speed(original_model, optimized_model, input_tensor, iterations=1000):
    import time
    
    # Warm up
    for _ in range(10):
        _ = original_model(input_tensor)
        _ = optimized_model(input_tensor)
    
    # Time original model
    start = time.time()
    for _ in range(iterations):
        _ = original_model(input_tensor)
    original_time = time.time() - start
    
    # Time optimized model
    start = time.time()
    for _ in range(iterations):
        _ = optimized_model(input_tensor)
    optimized_time = time.time() - start
    
    speedup = original_time / optimized_time
    return original_time, optimized_time, speedup
```
Model optimization techniques provide:
A) Faster inference speed
B) Reduced memory usage
C) Better deployment efficiency
D) All of the above

**Question 100:** Advanced custom CUDA memory management:
```python
def efficient_memory_management():
    # Check memory status
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Custom memory management
    with torch.cuda.device(0):
        # Allocate tensors
        large_tensor = torch.randn(10000, 10000, device='cuda')
        
        # Process in chunks to avoid memory spike
        def process_chunks(tensor, chunk_size=1000):
            results = []
            for i in range(0, tensor.size(0), chunk_size):
                chunk = tensor[i:i+chunk_size]
                result = expensive_operation(chunk)
                results.append(result.cpu())  # Move to CPU immediately
                del result  # Explicit deletion
                torch.cuda.empty_cache()  # Clear cache
            return torch.cat(results).cuda()
        
        processed = process_chunks(large_tensor)
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
    
    return processed
```
This memory management strategy:
A) Prevents out-of-memory errors
B) Optimizes GPU memory usage
C) Enables processing larger datasets
D) All of the above

---

## **ANSWERS**

**Question 1:** B) `torch.zeros(3, 4)`
**Question 2:** B) `torch.tensor()`
**Question 3:** A) `x.size()`
**Question 4:** B) `torch.ones(2, 3)`
**Question 5:** B) `tensor.cuda()`
**Question 6:** C) `a * b`
**Question 7:** B) `torch.arange(10)`
**Question 8:** B) Normal (0, 1)
**Question 9:** A) `x.reshape(3, 4)`
**Question 10:** A) `tensor.cpu()`
**Question 11:** C) `x[1, 2]`
**Question 12:** B) `torch.sum()`
**Question 13:** B) `torch.eye(4)`
**Question 14:** A) `requires_grad=True`
**Question 15:** B) `x.clone()`
**Question 16:** B) (3, 4)
**Question 17:** D) Both B and C are correct
**Question 18:** C) `view()` requires the tensor to be contiguous
**Question 19:** A) `torch.cat([a, b], dim=0)`
**Question 20:** B) Remove dimensions of size 1
**Question 21:** B) Base class for all neural network modules
**Question 22:** B) `nn.Linear(10, 5)`
**Question 23:** A) `max(0, x)`
**Question 24:** B) `loss.backward()`
**Question 25:** A) `torch.optim.SGD(model.parameters(), lr=0.01)`
**Question 26:** B) `optimizer.zero_grad()`
**Question 27:** B) Disable gradient computation
**Question 28:** B) 32x32
**Question 29:** A) 1/2
**Question 30:** B) `torch.save(model.state_dict(), 'model.pth')`
**Question 31:** B) Create batches and shuffle data
**Question 32:** A) `model.eval()`
**Question 33:** C) LogSoftmax + NLL Loss
**Question 34:** D) Both A and B are correct
**Question 35:** A) Before activation function
**Question 36:** A) Get elements at (0,0), (1,2), (2,1)
**Question 37:** C) Both performance and some operations
**Question 38:** C) `__init__`, `__len__` and `__getitem__`
**Question 39:** C) Prevent exploding gradients
**Question 40:** B) Define custom gradient computation
**Question 41:** A) `softmax(QK^T/√d)V`
**Question 42:** C) Combination of float16 and float32
**Question 43:** B) Graph is created at runtime
**Question 44:** B) `output = layer(input) + input`
**Question 45:** A) Data parallelism
**Question 46:** C) Compile model to TorchScript
**Question 47:** C) Cosine annealing
**Question 48:** D) Both A and B are possible
**Question 49:** D) All of the above
**Question 50:** D) All of the above
**Question 51:** C) (3,)
**Question 52:** A) (32, 128, 256)
**Question 53:** A) Gradient checkpointing
**Question 54:** A) (10, 3, 5)
**Question 55:** D) All of the above
**Question 56:** A) `batched_matrix_inverse` - vectorized operations
**Question 57:** D) Both B and C are correct
**Question 58:** C) `result.dtype` is `torch.float64`
**Question 59:** B) To maintain equivalent learning dynamics
**Question 60:** A) Method 1 (`torch.cat`)
**Question 61:** B) (50, 33)
**Question 62:** D) Both A and C
**Question 63:** A) `[[0, 0, 3], [4, 0, 5]]`
**Question 64:** D) All of the above
**Question 65:** D) All of the above
**Question 66:** A) Shape: (2, 2, 2), contains elements with specific stride pattern
**Question 67:** B) Option B - optimized function
**Question 68:** C) Manual load balancing
**Question 69:** D) All of the above
**Question 70:** D) All of the above
**Question 71:** B) Method 2 - direct boolean conversion
**Question 72:** A) Warmup + step decay
**Question 73:** D) All of the above
**Question 74:** A) (3, 5, 4, 6)
**Question 75:** D) All of the above
**Question 76:** B) Only B works (reshape is more flexible)
**Question 77:** D) All of the above
**Question 78:** D) All of the above
**Question 79:** D) All of the above
**Question 80:** D) All of the above
**Question 81:** D) All of the above
**Question 82:** D) All of the above
**Question 83:** D) All of the above
**Question 84:** D) All of the above
**Question 85:** D) All of the above
**Question 86:** D) All of the above
**Question 87:** D) All of the above
**Question 88:** D) All of the above
**Question 89:** D) All of the above
**Question 90:** D) All of the above
**Question 91:** D) All of the above
**Question 92:** D) All of the above
**Question 93:** D) All of the above
**Question 94:** D) All of the above
**Question 95:** D) All of the above
**Question 96:** D) All of the above
**Question 97:** D) All of the above
**Question 98:** D) All of the above
**Question 99:** D) All of the above
**Question 100:** D) All of the above 