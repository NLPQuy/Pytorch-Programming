# Phase 1: PyTorch Fundamentals - Complete Mastery Program
**Duration**: Weeks 1-6 (6 weeks)  
**Focus**: Master PyTorch syntax + Apply to logical problem solving  
**Methodology**: 65% repetitive practice (muscle memory) + 35% thinking applications  
**Goal**: Fluent PyTorch coding without documentation + problem-solving skills

---

## Week 1: Tensor Operations - Foundation Building

### Learning Objectives
- Memorize essential tensor creation and manipulation syntax
- Apply tensor operations to solve mathematical problems
- Build muscle memory for common PyTorch patterns
- Develop intuitive understanding of tensor operations

---

### Exercise 1.1: Tensor Creation Syntax Drill
**Type**: Muscle Memory  
**Time**: 2-3 hours  
**Difficulty**: Beginner

#### Objective
Memorize all basic tensor creation methods through repetitive coding.

#### Requirements
Write each tensor creation method EXACTLY 10 times following these specific requirements:

**TEMPLATE PATTERNS TO FOLLOW:**

```python
import torch

# Template for torch.zeros():
zeros_X = torch.zeros(shape_parameters)  # comment describing purpose

# Template for torch.ones():
ones_X = torch.ones(shape_parameters)    # comment describing purpose

# Template for torch.rand():
rand_X = torch.rand(shape_parameters)    # comment describing purpose

# Template for torch.randn():
randn_X = torch.randn(shape_parameters)  # comment describing purpose

# Template for torch.arange():
arange_X = torch.arange(start, stop, step)  # comment describing purpose

# Template for torch.linspace():
linspace_X = torch.linspace(start, end, steps)  # comment describing purpose

# Template for torch.tensor():
tensor_X = torch.tensor(data_structure)  # comment describing purpose

# Template for torch.empty():
empty_X = torch.empty(shape_parameters)  # comment describing purpose
```

**Practice Set 1: torch.zeros() - 10 REQUIRED variants**
- Variant 1: Create zeros tensor with 2D shape (3,4)  
- Variant 2: Create zeros tensor with 3D shape (2,3,4)
- Variant 3: Create zeros tensor with 1D shape (5 elements)
- Variant 4: Create zeros tensor with 2D row vector shape (1,10)
- Variant 5: Create zeros tensor with 2D column vector shape (10,1)
- Variant 6: Create zeros tensor with 4D shape (2,2,2,2)
- Variant 7: Create zeros tensor with shape (3,4) and dtype=torch.float32
- Variant 8: Create zeros tensor with shape (5,5) and dtype=torch.int64
- Variant 9: Create zeros tensor with single element (shape 1)
- Variant 10: Create zeros tensor with square matrix shape (8,8)

**Practice Set 2: torch.ones() - 10 REQUIRED variants**
- Variant 1: Create ones tensor with 2D shape (3,4)
- Variant 2: Create ones tensor with 3D shape (2,3,4)  
- Variant 3: Create ones tensor with 1D shape (5 elements)
- Variant 4: Create ones tensor with 2D row vector shape (1,10)
- Variant 5: Create ones tensor with 2D column vector shape (10,1)
- Variant 6: Create ones tensor with 4D shape (2,2,2,2)
- Variant 7: Create ones tensor with shape (3,4) and dtype=torch.float32
- Variant 8: Create ones tensor with shape (5,5) and dtype=torch.int64
- Variant 9: Create ones tensor with single element (shape 1)
- Variant 10: Create ones tensor with square matrix shape (6,6)

**Practice Set 3: torch.rand() - 10 REQUIRED variants**
- Variant 1: Create random tensor with shape 2D (3,4)
- Variant 2: Create random tensor with shape 3D (2,3,4)
- Variant 3: Create random tensor with shape 1D (100 elements)
- Variant 4: Create random tensor with shape 2D row vector (1,5)
- Variant 5: Create random tensor with shape 2D column vector (5,1)
- Variant 6: Create random tensor with shape 3D cubic (2,2,2)
- Variant 7: Create random tensor with square shape (10,10)
- Variant 8: Create random tensor with shape 4D (1,1,1,1)
- Variant 9: Create random tensor with shape 1D (50 elements)
- Variant 10: Create random tensor with shape 4D (3,3,3,3)

**Practice Set 4: torch.randn() - 10 REQUIRED variants**
- Variant 1: Create randn tensor with shape 2D (3,4)
- Variant 2: Create randn tensor with shape 3D (2,3,4)
- Variant 3: Create randn tensor with shape 1D (100 elements)
- Variant 4: Create randn tensor with shape 2D row vector (1,5)
- Variant 5: Create randn tensor with shape 2D column vector (5,1)
- Variant 6: Create randn tensor with shape 3D cubic (2,2,2)
- Variant 7: Create randn tensor with square shape (10,10)
- Variant 8: Create randn tensor with shape 4D (1,1,1,1)
- Variant 9: Create randn tensor with shape 1D (50 elements)
- Variant 10: Create randn tensor with shape 4D (3,3,3,3)

**Practice Set 5: torch.arange() - 10 REQUIRED variants**
- Variant 1: Create sequence from 0 to 9 (10 elements)
- Variant 2: Create sequence from 1 to 9 (9 elements)
- Variant 3: Create sequence from 0 to 9 with step=2
- Variant 4: Create sequence from 5 to 49 with step=5
- Variant 5: Create sequence from -10 to 9 (20 elements)
- Variant 6: Create sequence from 100 down to 1 with step=-1
- Variant 7: Create sequence from 0 to 0.9 with step=0.1 (float)
- Variant 8: Create sequence from 1.0 to 9.5 with step=0.5 (float)
- Variant 9: Create sequence from 0 to 90 with step=10
- Variant 10: Create sequence from -5 to 4 (symmetric range)

**Practice Set 6: torch.linspace() - 10 REQUIRED variants**
- Variant 1: Create 5 evenly spaced points from 0 to 1
- Variant 2: Create 10 evenly spaced points from -1 to 1
- Variant 3: Create 50 evenly spaced points from 0 to 100
- Variant 4: Create 100 evenly spaced points from 0 to 1 (fine resolution)
- Variant 5: Create 21 evenly spaced points from -10 to 10
- Variant 6: Create 100 evenly spaced points from 0 to 2π
- Variant 7: Create 11 evenly spaced points from 1 to 2
- Variant 8: Create 200 evenly spaced points from -100 to 100
- Variant 9: Create 2 points from 0 to 1 (minimum points)
- Variant 10: Create 20 evenly spaced points from 0.5 to 1.5

**Practice Set 7: torch.tensor() - 10 REQUIRED variants**
- Variant 1: Create a tensor from a 1D integer list [1, 2, 3]
- Variant 2: Create a tensor from a 2D nested integer list [[1, 2], [3, 4]]
- Variant 3: Convert a NumPy array np.array([1.0, 2.0, 3.0]) to a tensor
- Variant 4: Convert a 2D float matrix (2×3) from a NumPy array to a tensor
- Variant 5: Create a boolean tensor from the list [True, False, True] and convert it to a NumPy array
- Variant 6: Create a 3D tensor from a nested list with shape (2×2×2), then convert it to a NumPy array
- Variant 7: Convert a NumPy array with a single element np.array([1]) to a tensor
- Variant 8: Create a tensor from a longer 1D list containing 10 elements
- Variant 9: Create a tensor from a 3×3 identity matrix generated using NumPy (np.eye(3))
- Variant 10: Create a tensor from a 2D list of negative integers [[-1, -2], [-3, -4]] and convert it to a NumPy array

**Practice Set 8: torch.empty() - 10 REQUIRED variants**
- Variant 1: Create empty tensor with shape 2D (3,4)
- Variant 2: Create empty tensor with shape 3D (2,3,4)
- Variant 3: Create empty tensor with shape 1D (5 elements)
- Variant 4: Create empty tensor with shape 2D row vector (1,10)
- Variant 5: Create empty tensor with shape 2D column vector (10,1)
- Variant 6: Create empty tensor with shape 4D (2,2,2,2)
- Variant 7: Create empty tensor with shape (3,4) and dtype=torch.float32
- Variant 8: Create empty tensor with shape (5,5) and dtype=torch.int64
- Variant 9: Create empty tensor with single element (shape 1)
- Variant 10: Create empty tensor with square matrix shape (7,7)

**CRITICAL**: You must write ALL 80 lines of code (10 × 8 methods). No shortcuts allowed!

#### Deliverables
- **tensor_creation_drill.py**: 80 lines of tensor creation code (10 variations × 8 methods)
- Practice daily for 1 week until you can write any tensor creation method without looking

#### Self-Check Questions (answer without looking at code)
1. How do you create a 3×4 tensor of zeros?
2. How do you create random numbers between 0 and 1?
3. How do you create a sequence from 0 to 9?
4. How do you create 5 evenly spaced numbers between 0 and 1?

---

### Exercise 1.2: Tensor Indexing & Slicing Syntax Drill
**Type**: Muscle Memory  
**Time**: 2-3 hours  
**Difficulty**: Beginner

#### Objective
Memorize tensor indexing and slicing syntax through repetitive practice.

#### Requirements
Practice each indexing pattern **EXACTLY 15 times** with these specific requirements:

**Base setup**: Create 3D tensor with shape (4, 5, 6) using torch.randn()

**TEMPLATE PATTERNS TO FOLLOW:**

```python
import torch

# Base tensor setup
x = torch.randn(4, 5, 6)  # 3D tensor for practice

# Template for Single element access:
elem_X = x[index1, index2, index3]  # comment describing position

# Template for First dimension slicing:
slice_X = x[index]  # comment describing which page

# Template for Range slicing:
range_X = x[start:end]  # comment describing range

# Template for All dimensions slicing:
all_X = x[dim1_spec, dim2_spec, dim3_spec]  # comment describing pattern

# Template for Step slicing:
step_X = x[start:end:step]  # or x[:, start:end:step] # comment describing step

# Template for Negative indexing:
neg_X = x[negative_indices]  # comment describing negative pattern
```

**Practice Set 1: Single element access - 15 REQUIRED variants**
- Variant 1: Access element at position [0, 1, 2]
- Variant 2: Access element at position [1, 0, 3]  
- Variant 3: Access element at position [2, 2, 1]
- Variant 4: Access element at position [3, 4, 5] (last valid indices)
- Variant 5: Access first element [0, 0, 0]
- Variant 6: Access last element [-1, -1, -1] (all negative)
- Variant 7: Access element with mixed indices [-1, 0, 0]
- Variant 8: Access element with negative in middle dimension [0, -1, 0]
- Variant 9: Access element with negative in last dimension [0, 0, -1]
- Variant 10: Access element at position [1, 2, 3]
- Variant 11: Access element at position [2, 1, 4]
- Variant 12: Access element at position [0, 3, 2]
- Variant 13: Access element at position [3, 1, 1]
- Variant 14: Access element at position [1, 4, 0]
- Variant 15: Access element at position [2, 3, 5]

**Practice Set 2: First dimension slicing - 15 REQUIRED variants**
- Variant 1: Get first page [0]
- Variant 2: Get second page [1]
- Variant 3: Get third page [2]
- Variant 4: Get fourth page [3]
- Variant 5: Get last page [-1] (negative indexing)
- Variant 6: Get third from last page [-2]
- Variant 7: Get second from last page [-3]
- Variant 8: Get first page using negative [-4]
- Variant 9: Create new tensor with 6 pages, get first page
- Variant 10: From 6-page tensor, get last page [5]
- Variant 11: From 6-page tensor, get last page [-1]
- Variant 12: From 6-page tensor, get middle page [2]
- Variant 13: From 6-page tensor, get page [3]
- Variant 14: From 6-page tensor, get page [4]
- Variant 15: From 6-page tensor, get page [1]

**Practice Set 3: Range slicing - 15 REQUIRED variants**
- Variant 1: Get first 2 pages [0:2]
- Variant 2: Get middle 2 pages [1:3]
- Variant 3: Get last 2 pages [2:4]
- Variant 4: Get first 2 pages with implicit start [:2]
- Variant 5: Get last 2 pages with implicit end [2:]
- Variant 6: Get all pages [:]
- Variant 7: Get all pages explicit [0:4]
- Variant 8: Skip first page [1:4]
- Variant 9: Skip last page [0:3]
- Variant 10: Get 1 page as range [0:1]
- Variant 11: Get last page as range [3:4]
- Variant 12: Get last 2 pages using negative [-2:]
- Variant 13: Get all except last page [:-1]
- Variant 14: Range with negative indices [-3:-1]
- Variant 15: Remove first and last page [1:-1]

**Practice Set 4: All dimensions slicing - 15 REQUIRED variants**
- Variant 1: First page, all rows, all columns [0, :, :]
- Variant 2: Second page, all rows, all columns [1, :, :]
- Variant 3: All pages, first row, all columns [:, 0, :]
- Variant 4: All pages, second row, all columns [:, 1, :]
- Variant 5: All pages, all rows, first column [:, :, 0]
- Variant 6: All pages, all rows, second column [:, :, 1]
- Variant 7: Last page, all rows, all columns [-1, :, :]
- Variant 8: All pages, last row, all columns [:, -1, :]
- Variant 9: All pages, all rows, last column [:, :, -1]
- Variant 10: First page, first row, all columns [0, 0, :]
- Variant 11: First page, all rows, first column [0, :, 0]
- Variant 12: All pages, first row, first column [:, 0, 0]
- Variant 13: Middle page, all rows, all columns [2, :, :]
- Variant 14: All pages, middle row, all columns [:, 2, :]
- Variant 15: All pages, all rows, middle column [:, :, 3]

**Practice Set 5: Step slicing - 15 REQUIRED variants**
- Variant 1: Every second page [::2]
- Variant 2: Every page (step 1) [::1]
- Variant 3: Every second row [:, ::2]
- Variant 4: Every row (step 1) [:, ::1]
- Variant 5: Every second column [:, :, ::2]
- Variant 6: Every column (step 1) [:, :, ::1]
- Variant 7: Every second page starting from index 1 [1::2]
- Variant 8: Every second row starting from index 1 [:, 1::2]
- Variant 9: Every second column starting from index 1 [:, :, 1::2]
- Variant 10: Every third page [::3]
- Variant 11: Every third row [:, ::3]
- Variant 12: Every third column [:, :, ::3]
- Variant 13: Pages 0, 2 with explicit range [0:4:2]
- Variant 14: Rows 0, 2, 4 [:, 0:5:2]
- Variant 15: Columns 0, 2, 4 [:, :, 0:6:2]

**Practice Set 6: Negative indexing - 15 REQUIRED variants**
- Variant 1: Last page [-1]
- Variant 2: Second from last page [-2]
- Variant 3: Last row of all pages [:, -1]
- Variant 4: Second from last row [:, -2]
- Variant 5: Last column of all pages [:, :, -1]
- Variant 6: Second from last column [:, :, -2]
- Variant 7: Last page, last row [-1, -1]
- Variant 8: Last page, all rows, last column [-1, :, -1]
- Variant 9: All pages, last row, last column [:, -1, -1]
- Variant 10: Last element [-1, -1, -1]
- Variant 11: Second from last element in each dimension [-2, -2, -2]
- Variant 12: First page, last row, all columns [0, -1, :]
- Variant 13: Last page, first row, all columns [-1, 0, :]
- Variant 14: All pages, first row, last column [:, 0, -1]
- Variant 15: Mixed positive/negative indices [1, -2, 3]

**CRITICAL**: You must write ALL 90 lines of code (15 × 6 patterns). Each pattern teaches different indexing skills!

#### Deliverables
- **indexing_drill.py**: 90 lines of indexing code (15 variations × 6 patterns)
- **indexing_quiz.py**: Self-quiz to test syntax memory

#### Daily Practice Routine
- Day 1-2: Basic indexing
- Day 3-4: Range slicing
- Day 5-6: Advanced patterns
- Day 7: Mixed practice without looking

---

### Exercise 1.3: Mathematical Problem Solving Applications
**Type**: Thinking + Application  
**Time**: 3-4 hours  
**Difficulty**: Beginner-Intermediate

#### Objective
Apply memorized tensor syntax to solve real mathematical problems.

#### Requirements

**Problem 1: Linear System Solver**
```python
import torch

def solve_linear_system():
    """
    Solve the system: Ax = b where
    A = [[2, 1], [1, 3]]
    b = [5, 7]
    
    Expected solution: x = [1, 3]
    Use tensor operations to verify and solve.
    """
    # TODO: Create coefficient matrix A and vector b using tensor syntax
    # TODO: Solve using torch.linalg.solve()
    # TODO: Verify solution by computing A @ x and comparing with b
    # TODO: Handle edge cases (singular matrices)
    
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([5.0, 7.0])
    
    # Your implementation here
    # solution = ...
    # verification = ...
    # print results
    pass

def solve_3x3_system():
    """
    Solve 3x3 system: 3x + 2y - z = 1, x - y + 2z = 4, 2x + y + z = 3
    """
    # TODO: Set up 3x3 coefficient matrix
    # TODO: Solve and verify
    pass
```

**Problem 2: Polynomial Evaluator**
```python
def create_polynomial_coefficients():
    """
    Create coefficient tensors for polynomial: 3x³ + 2x² - x + 5
    Then evaluate at points x = [0, 1, 2, 3, 4]
    """
    # TODO: Create coefficient tensor [3, 2, -1, 5]
    # TODO: Create evaluation points tensor
    # TODO: Use tensor operations to evaluate polynomial
    # TODO: Compare with manual calculation
    
    coeffs = torch.tensor([3.0, 2.0, -1.0, 5.0])  # highest degree first
    x_vals = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    
    # Method 1: Using polynomial evaluation
    def evaluate_polynomial(x, coefficients):
        """Evaluate polynomial using tensor operations"""
        # TODO: Implement using powers and broadcasting
        pass
    
    # Method 2: Using Horner's method
    def horner_evaluation(x, coefficients):
        """More numerically stable evaluation"""
        # TODO: Implement Horner's method
        pass
    
    # Test both methods
    pass

def polynomial_operations():
    """
    Additional polynomial operations:
    - Find derivative coefficients
    - Evaluate derivative at points
    - Find approximate roots
    """
    # TODO: Implement derivative calculation
    # TODO: Root finding using tensor operations
    pass
```

**Problem 3: Matrix Transformations**
```python
def matrix_transformations():
    """
    Apply geometric transformations using matrix operations:
    1. Rotation matrix for 45 degrees
    2. Scaling matrix (2x in x-direction, 0.5x in y-direction)
    3. Apply to set of 2D points
    """
    import math
    
    def create_rotation_matrix(angle_degrees):
        """Create 2D rotation matrix"""
        # TODO: Convert degrees to radians
        # TODO: Create rotation matrix using cos/sin
        # TODO: Use torch.cos() and torch.sin()
        
        angle_rad = math.radians(angle_degrees)
        # R = [[cos(θ), -sin(θ)],
        #      [sin(θ),  cos(θ)]]
        pass
    
    def create_scaling_matrix(scale_x, scale_y):
        """Create 2D scaling matrix"""
        # TODO: Create scaling matrix
        # S = [[scale_x, 0],
        #      [0, scale_y]]
        pass
    
    def apply_transformation(points, transform_matrix):
        """Apply transformation to set of 2D points"""
        # TODO: Apply matrix multiplication
        # points shape: (N, 2), transform shape: (2, 2)
        # result = points @ transform_matrix.T
        pass
    
    # Test transformations
    # Create unit square points
    unit_square = torch.tensor([[0.0, 0.0], [1.0, 0.0], 
                               [1.0, 1.0], [0.0, 1.0]])
    
    # TODO: Apply 45-degree rotation
    # TODO: Apply scaling (2x, 0.5x)
    # TODO: Combine transformations
    # TODO: Visualize results (optional with matplotlib)
    
    pass
```

**Problem 4: Statistical Analysis**
```python
def statistical_analysis():
    """
    Given temperature data for a week, perform statistical analysis:
    - Calculate mean, median, std deviation
    - Find outliers (values > 2 std from mean)
    - Normalize data to z-scores
    """
    # Sample temperature data
    temperatures = torch.tensor([22.5, 25.1, 23.8, 28.2, 26.5, 24.1, 23.9])
    
    # TODO: Calculate statistics using tensor operations
    def calculate_statistics(data):
        """Calculate comprehensive statistics"""
        # TODO: mean = torch.mean(data)
        # TODO: std = torch.std(data)
        # TODO: median = torch.median(data)
        # TODO: min_val = torch.min(data)
        # TODO: max_val = torch.max(data)
        
        # Return as dictionary
        pass
    
    # TODO: Identify outliers using boolean indexing
    def find_outliers(data, threshold=2.0):
        """Find outliers using z-score method"""
        # TODO: Calculate z-scores
        # TODO: Find values where |z-score| > threshold
        # TODO: Use boolean indexing to extract outliers
        pass
    
    # TODO: Normalize data to z-scores
    def normalize_to_z_scores(data):
        """Normalize data to z-scores"""
        # TODO: z = (x - mean) / std
        pass
    
    # TODO: Create summary report
    def create_summary_report(data):
        """Generate comprehensive analysis report"""
        stats = calculate_statistics(data)
        outliers = find_outliers(data)
        normalized = normalize_to_z_scores(data)
        
        # TODO: Print formatted report
        pass
    
    # Run analysis
    create_summary_report(temperatures)

def batch_statistics():
    """
    Calculate statistics for multiple datasets simultaneously using broadcasting:
    - 5 different weeks of temperature data
    - Calculate mean/std for each week
    - Find which week had most stable temperature
    """
    # TODO: Create 2D tensor (5 weeks × 7 days)
    # TODO: Use dim parameter for statistics
    # TODO: Compare weeks using tensor operations
    
    # Create sample data: 5 weeks of temperature
    temp_data = torch.randn(5, 7) * 3 + 25  # 5 weeks, 7 days each
    
    def analyze_multiple_weeks(data):
        """Analyze temperature data for multiple weeks"""
        # TODO: Calculate mean for each week (dim=1)
        # TODO: Calculate std for each week (dim=1)
        # TODO: Find most stable week (lowest std)
        # TODO: Find warmest/coldest weeks
        pass
    
    pass
```

#### Evaluation Criteria
- [ ] Uses appropriate tensor creation methods
- [ ] Applies correct mathematical operations
- [ ] Handles edge cases properly
- [ ] Code is readable and well-commented

---

## Week 2: Shape Manipulation & Broadcasting

### Learning Objectives
- Memorize tensor reshaping syntax through repetitive practice
- Understand broadcasting rules through systematic examples
- Apply shape manipulation to solve data processing problems

---

### Exercise 2.1: Shape Manipulation Syntax Drill
**Type**: Muscle Memory  
**Time**: 3-4 hours  
**Difficulty**: Beginner

#### Objective
Memorize all tensor reshaping operations through repetition.

#### Requirements
Practice each reshaping method 15 times:

**Base setup**: Create base tensor with shape (2, 3, 4) using torch.randn()

**TEMPLATE PATTERNS TO FOLLOW:**

```python
import torch

# Base tensor setup
x = torch.randn(2, 3, 4)  # Original shape: (2, 3, 4)

# Template for view() operations:
view_X = x.view(new_shape)  # comment describing shape change

# Template for reshape() operations:
reshape_X = x.reshape(new_shape)  # comment describing reshape

# Template for unsqueeze() operations:
unsqueeze_X = x.unsqueeze(dim_position)  # comment describing added dimension

# Template for squeeze() operations (need tensor with singleton dims):
y = torch.randn(1, 2, 1, 4)  # tensor with singleton dimensions
squeeze_X = y.squeeze(dim_position)  # comment describing removed dimension

# Template for transpose() operations:
transpose_X = x.transpose(dim1, dim2)  # comment describing swapped dimensions

# Template for permute() operations:
permute_X = x.permute(dim_order)  # comment describing reordered dimensions

# Template for flatten() operations:
flatten_X = x.flatten(start_dim, end_dim)  # comment describing flatten range
```

**Practice Set 1: view() operations - 15 REQUIRED variants**
- Variant 1: Reshape tensor from (2,3,4) to (6,4)
- Variant 2: Reshape tensor from (2,3,4) to (2,12)
- Variant 3: Flatten tensor to 1D (24 elements)
- Variant 4: Reshape with auto-calculate first dimension (-1,4)
- Variant 5: Reshape with auto-calculate second dimension (2,-1)
- Variant 6: Reshape to (3,8)
- Variant 7: Reshape to (4,6)  
- Variant 8: Reshape to (1,24) (single row)
- Variant 9: Reshape to (24,1) (single column)
- Variant 10: Reshape to (12,2)
- Variant 11: Reshape to (8,3)
- Variant 12: Reshape to (1,2,12) (add dimension)
- Variant 13: Reshape to (2,1,12) (add dimension)
- Variant 14: Reshape to (6,2,2) (3D output)
- Variant 15: Reshape to (4,3,2) (3D output)

**Practice Set 2: reshape() operations - 15 REQUIRED variants**
- Variant 1: Reshape tensor from (2,3,4) to (6,4) using reshape()
- Variant 2: Reshape tensor from (2,3,4) to (2,12) using reshape()
- Variant 3: Flatten tensor to 1D using reshape()
- Variant 4: Reshape with auto-calculate (-1,4) using reshape()
- Variant 5: Reshape with auto-calculate (2,-1) using reshape()
- Variant 6: Reshape to (3,8) using reshape()
- Variant 7: Reshape to (4,6) using reshape()
- Variant 8: Reshape to (1,24) using reshape()
- Variant 9: Reshape to (24,1) using reshape()
- Variant 10: Reshape to (12,2) using reshape()
- Variant 11: Reshape to (8,3) using reshape()
- Variant 12: Reshape to (1,2,12) using reshape()
- Variant 13: Reshape to (2,1,12) using reshape()
- Variant 14: Reshape to (6,2,2) using reshape()
- Variant 15: Reshape to (4,3,2) using reshape()

**Practice Set 3: unsqueeze() operations - 15 REQUIRED variants**
- Variant 1: Add dimension at position 0: (2,3,4) → (1,2,3,4)
- Variant 2: Add dimension at position 1: (2,3,4) → (2,1,3,4)
- Variant 3: Add dimension at position 2: (2,3,4) → (2,3,1,4)
- Variant 4: Add dimension at position 3: (2,3,4) → (2,3,4,1)
- Variant 5: Add dimension at position -1: (2,3,4) → (2,3,4,1)
- Variant 6: Add dimension at position -2: (2,3,4) → (2,3,1,4)
- Variant 7: Add dimension at position -3: (2,3,4) → (2,1,3,4)
- Variant 8: Add dimension at position -4: (2,3,4) → (1,2,3,4)
- Variant 9: Create 2D tensor (3,4), add dimension at position 0
- Variant 10: From 2D tensor (3,4), add dimension at position 1
- Variant 11: From 2D tensor (3,4), add dimension at position 2
- Variant 12: From 1D tensor (5), add dimension at position 0
- Variant 13: From 1D tensor (5), add dimension at position 1
- Variant 14: From 1D tensor (5), add dimension at position -1
- Variant 15: From 1D tensor (5), add dimension at position -2

**Practice Set 4: squeeze() operations - 15 REQUIRED variants**
**Base setup**: Create tensor with singleton dimensions: shape (1, 2, 1, 4)
- Variant 1: Remove all singleton dimensions
- Variant 2: Remove specific dimension at position 0
- Variant 3: Remove specific dimension at position 2
- Variant 4: Create tensor (1,3,1,1,5), remove all singleton dims
- Variant 5: From tensor (1,3,1,1,5), remove dim at position 0
- Variant 6: From tensor (1,3,1,1,5), remove dim at position 2
- Variant 7: From tensor (1,3,1,1,5), remove dim at position 3
- Variant 8: Create tensor (1,1,4), remove all singleton dims
- Variant 9: From tensor (1,1,4), remove dim at position 0
- Variant 10: From tensor (1,1,4), remove dim at position 1
- Variant 11: Create tensor (2,1,3,1), remove all singleton dims
- Variant 12: From tensor (2,1,3,1), remove dim at position 1
- Variant 13: From tensor (2,1,3,1), remove dim at position 3
- Variant 14: Create tensor (1,5,1), remove specific dimensions
- Variant 15: Practice with multiple different singleton dimensions

**Practice Set 5: transpose() operations - 15 REQUIRED variants**
- Variant 1: Swap dimensions 0 and 1: (2,3,4) → (3,2,4)
- Variant 2: Swap dimensions 1 and 2: (2,3,4) → (2,4,3)
- Variant 3: Swap dimensions 0 and 2: (2,3,4) → (4,3,2)
- Variant 4: Create tensor (5,6,7), swap dims 0,1
- Variant 5: From tensor (5,6,7), swap dims 1,2
- Variant 6: From tensor (5,6,7), swap dims 0,2
- Variant 7: Create 2D tensor (4,5), transpose dims 0,1
- Variant 8: Create 4D tensor (2,3,4,5), swap dims 0,1
- Variant 9: From 4D tensor (2,3,4,5), swap dims 1,2
- Variant 10: From 4D tensor (2,3,4,5), swap dims 2,3
- Variant 11: From 4D tensor (2,3,4,5), swap dims 0,3
- Variant 12: Practice with negative indexing: swap dims -1,-2
- Variant 13: Practice with negative indexing: swap dims 0,-1
- Variant 14: Practice with negative indexing: swap dims -2,-3
- Variant 15: Practice multiple consecutive transpose operations

**Practice Set 6: permute() operations - 15 REQUIRED variants**
- Variant 1: Reorder dimensions (2,3,4) → (4,2,3) using (2,0,1)
- Variant 2: Reorder dimensions (2,3,4) → (3,4,2) using (1,2,0)
- Variant 3: Reverse dimension order (2,3,4) → (4,3,2) using (2,1,0)
- Variant 4: Create tensor (5,6,7), permute with (2,0,1)
- Variant 5: From tensor (5,6,7), permute with (1,2,0)
- Variant 6: From tensor (5,6,7), permute with (2,1,0)
- Variant 7: From tensor (5,6,7), permute with (0,2,1)
- Variant 8: Create 4D tensor (2,3,4,5), permute with (3,2,1,0)
- Variant 9: From 4D tensor (2,3,4,5), permute with (1,0,2,3)
- Variant 10: From 4D tensor (2,3,4,5), permute with (0,2,1,3)
- Variant 11: From 4D tensor (2,3,4,5), permute with (3,0,1,2)
- Variant 12: Practice with permute to convert from NHWC to NCHW format
- Variant 13: Practice with permute to convert from NCHW to NHWC format
- Variant 14: Practice with complex permutations for 5D tensor
- Variant 15: Practice with custom dimension ordering

**Practice Set 7: flatten() operations - 15 REQUIRED variants**
- Variant 1: Flatten entire tensor (2,3,4) → (24,)
- Variant 2: Flatten from dimension 1: (2,3,4) → (2,12)
- Variant 3: Flatten to dimension 1: (2,3,4) → (6,4)
- Variant 4: Create tensor (2,3,4,5), flatten entire
- Variant 5: From tensor (2,3,4,5), flatten from dim 1
- Variant 6: From tensor (2,3,4,5), flatten from dim 2
- Variant 7: From tensor (2,3,4,5), flatten from dim 3
- Variant 8: From tensor (2,3,4,5), flatten to dim 1
- Variant 9: From tensor (2,3,4,5), flatten to dim 2
- Variant 10: From tensor (2,3,4,5), flatten from dim 1 to dim 2
- Variant 11: From tensor (2,3,4,5), flatten from dim 2 to dim 3
- Variant 12: Practice with 5D tensor, flatten selective dimensions
- Variant 13: Practice with batch processing: preserve batch dim
- Variant 14: Practice with image tensors: flatten spatial dims only
- Variant 15: Practice with sequence data: flatten specific dimensions

**CRITICAL**: You must write ALL 105 lines of code (15 × 7 methods). Each method teaches different shape manipulation skills!

#### Deliverables
- **shape_manipulation_drill.py**: 75+ shape manipulation operations
- **shape_reference.py**: Quick reference of all reshaping methods

---

### Exercise 2.2: Broadcasting Rules Practice
**Type**: Understanding + Repetition  
**Time**: 3-4 hours  
**Difficulty**: Beginner-Intermediate

#### Objective
Understand PyTorch broadcasting through systematic repetitive examples.

#### Requirements
Practice broadcasting with **EXACTLY 20 different shape combinations** with specific requirements:

**TEMPLATE PATTERNS TO FOLLOW:**

```python
import torch

# Template for Simple Broadcasting:
a = torch.randn(shape1)  # comment describing tensor 1
b = torch.randn(shape2)  # comment describing tensor 2  
result = a + b           # comment describing expected shape
print(f"Broadcasting: {a.shape} + {b.shape} = {result.shape}")

# Template for Manual Broadcasting Understanding:
a_expanded = a.expand(target_shape)  # comment about expansion
b_expanded = b.expand(target_shape)  # comment about expansion
manual_result = a_expanded + b_expanded
auto_result = a + b
print(f"Manual vs Auto: {torch.allclose(manual_result, auto_result)}")

# Template for Error Handling:
try:
    result = tensor1 + tensor2
    print(f"Success: {result.shape}")
except RuntimeError as e:
    print(f"Broadcasting error: {e}")
```

**Practice Set 1: Simple Broadcasting - 8 REQUIRED variants**
- Variant 1: Vector + Scalar: (3,) + (1,) → (3,)
- Variant 2: Matrix + Vector: (3,4) + (4,) → (3,4)
- Variant 3: Matrix + Column Vector: (3,4) + (3,1) → (3,4)
- Variant 4: Larger Vector + Scalar: (10,) + (1,) → (10,)
- Variant 5: Different Matrix + Vector: (5,6) + (6,) → (5,6)
- Variant 6: Matrix + Column Vector khác: (4,5) + (4,1) → (4,5)
- Variant 7: Matrix + Row Vector: (3,4) + (1,4) → (3,4)
- Variant 8: Square Matrix + Vector: (5,5) + (5,) → (5,5)

**Practice Set 2: 3D Broadcasting - 6 REQUIRED variants**
- Variant 9: 3D + 2D: (2,3,4) + (3,4) → (2,3,4)
- Variant 10: 3D + 1D: (2,3,4) + (4,) → (2,3,4)
- Variant 11: 3D + Vector khác: (5,6,7) + (7,) → (5,6,7)
- Variant 12: 3D + 2D khác: (4,5,6) + (5,6) → (4,5,6)
- Variant 13: 3D + Column Vector: (2,3,4) + (3,1) → (2,3,4)
- Variant 14: 3D + Singleton 2D: (3,4,5) + (1,5) → (3,4,5)

**Practice Set 3: Complex Broadcasting with Singleton Dimensions - 6 REQUIRED variants**
- Variant 15: (2,1,4) + (1,3,1) → (2,3,4)
- Variant 16: (1,5,1) + (3,1,2) → (3,5,2)
- Variant 17: (4,1,6) + (1,2,1) → (4,2,6)
- Variant 18: (1,1,5) + (3,4,1) → (3,4,5)
- Variant 19: (2,3,1,1) + (1,1,4,5) → (2,3,4,5)
- Variant 20: (1,6,1,8) + (5,1,7,1) → (5,6,7,8)

**Additional Requirements for each variant:**
1. **Print shapes**: Print original shapes and resulting shape
2. **Verify result**: Check that operation actually works
3. **Test with multiple operations**: +, -, *, /
4. **Manual expansion practice**: Use .expand() to understand broadcasting
5. **Error handling**: Try incompatible shape combinations

**Manual Broadcasting Understanding Practice:**
- Variant 21: Manually expand (2,1,4) and (1,3,1) to (2,3,4) then perform operation
- Variant 22: Manually expand (1,5,1) and (3,1,2) to (3,5,2) then perform operation  
- Variant 23: Compare manual expansion with automatic broadcasting
- Variant 24: Try expansion with .expand() and verify compatibility
- Variant 25: Practice with .expand_as() method

**Daily Practice Requirements:**
- **Day 1**: Simple broadcasting (Variants 1-8) - 20 iterations each
- **Day 2**: 3D broadcasting (Variants 9-14) - 20 iterations each  
- **Day 3**: Complex broadcasting (Variants 15-20) - 15 iterations each
- **Day 4**: Manual expansion practice (Variants 21-25) - 10 iterations each
- **Day 5**: Mixed practice all variants
- **Day 6**: Error cases and edge cases
- **Day 7**: Speed practice without looking up documentation

**CRITICAL**: You must practice ALL 25 variants. Broadcasting is a critical foundation of PyTorch!

#### Deliverables
- **broadcasting_practice.py**: 100+ broadcasting examples
- **broadcasting_quiz.py**: Self-test for broadcasting understanding
- **broadcasting_cheatsheet.md**: Personal reference of broadcasting rules

---

### Exercise 2.3: Data Processing Applications
**Type**: Thinking + Application  
**Time**: 4-5 hours  
**Difficulty**: Intermediate

#### Objective
Apply shape manipulation to solve real data processing problems.

#### Requirements

**Problem 1: Image Batch Processing**
```python
def image_batch_processor():
    """
    Process a batch of images with different required transformations:
    
    Input: Batch of images (batch_size, channels, height, width)
    Tasks:
    1. Convert from (batch, height, width, channels) to PyTorch format
    2. Extract patches for analysis
    3. Reshape for different model requirements
    4. Handle variable batch sizes efficiently
    """
    
    # Simulate different image formats
    numpy_format = torch.randn(32, 224, 224, 3)  # HWC format
    pytorch_format = torch.randn(32, 3, 224, 224) # CHW format
    
    def convert_hwc_to_chw(images):
        """Convert from (B, H, W, C) to (B, C, H, W)"""
        # TODO: Use permute() to reorder dimensions
        # TODO: Verify the conversion is correct
        
        # Hint: permute(0, 3, 1, 2) moves channels to position 1
        converted = images.permute(0, 3, 1, 2)
        
        # Verification
        print(f"Original shape: {images.shape}")
        print(f"Converted shape: {converted.shape}")
        return converted
    
    def extract_image_patches(images, patch_size=16):
        """Extract non-overlapping patches from images"""
        # TODO: Reshape images to extract patches
        # TODO: Handle images that don't divide evenly
        # Input: (batch, channels, height, width)
        # Output: (batch, num_patches, channels, patch_size, patch_size)
        
        batch_size, channels, height, width = images.shape
        
        # Calculate number of patches
        patches_h = height // patch_size
        patches_w = width // patch_size
        
        # TODO: Reshape to extract patches
        # Step 1: Reshape to separate patch dimensions
        # Step 2: Permute to group patches together
        
        # Hint: Use view() and permute() operations
        pass
    
    def flatten_for_mlp(images):
        """Flatten images for MLP input while preserving batch dimension"""
        # TODO: Flatten spatial dimensions only
        # Input: (batch, channels, height, width)  
        # Output: (batch, channels * height * width)
        
        batch_size = images.shape[0]
        flattened = images.view(batch_size, -1)
        
        print(f"Original shape: {images.shape}")
        print(f"Flattened shape: {flattened.shape}")
        return flattened
    
    def create_sliding_windows(sequence, window_size, stride=1):
        """Create sliding windows from sequence data"""
        # TODO: Use unfold() or manual reshaping
        # Input: (batch, sequence_length, features)
        # Output: (batch, num_windows, window_size, features)
        
        # Method 1: Using unfold()
        windows = sequence.unfold(1, window_size, stride)
        # TODO: Reshape to desired output format
        
        # Method 2: Manual approach with indexing
        # TODO: Create windows using indexing and stacking
        
        pass
    
    # Test all functions
    test_images = torch.randn(4, 224, 224, 3)  # HWC format
    
    # Test conversions
    chw_images = convert_hwc_to_chw(test_images)
    flattened = flatten_for_mlp(chw_images)
    
    # TODO: Test patch extraction
    # TODO: Test with different batch sizes

def time_series_reshaping():
    """
    Reshape time series data for different analysis needs:
    
    Original: Daily temperature data for multiple cities over a year
    Shape: (num_cities, 365) 
    
    Reshape for:
    1. Weekly analysis: (num_cities, 52, 7)
    2. Monthly analysis: (num_cities, 12, ~30)
    3. Seasonal analysis: (num_cities, 4, ~91)
    """
    
    # Simulate temperature data
    num_cities = 10
    days_per_year = 365
    temperatures = torch.randn(num_cities, days_per_year) * 5 + 20  # 20°C ± 5°C
    
    def reshape_to_weeks(data):
        """Reshape to weekly format, handle remaining days"""
        # TODO: Reshape to (cities, weeks, days_per_week)
        # TODO: Handle the extra day in 365-day year
        
        # 365 days = 52 weeks + 1 extra day
        # Option 1: Truncate the extra day
        truncated_data = data[:, :364]  # Remove last day
        weekly_data = truncated_data.view(num_cities, 52, 7)
        
        # Option 2: Pad to make it divisible
        # TODO: Implement padding approach
        
        print(f"Original shape: {data.shape}")
        print(f"Weekly shape: {weekly_data.shape}")
        return weekly_data
    
    def reshape_to_months(data):
        """Reshape to monthly format with variable month lengths"""
        # TODO: Handle different month lengths (28, 30, 31 days)
        # TODO: Consider padding or truncation strategies
        
        # Approximate: 365 ÷ 12 ≈ 30.4 days per month
        # Option 1: Use 30 days per month (360 days total)
        truncated_data = data[:, :360]
        monthly_data = truncated_data.view(num_cities, 12, 30)
        
        # Option 2: Handle real month lengths
        # TODO: Create variable length months
        # This would require more complex reshaping or padding
        
        print(f"Monthly shape (approximated): {monthly_data.shape}")
        return monthly_data
    
    def seasonal_analysis(data):
        """Group data by seasons for climate analysis"""
        # TODO: Define seasons (e.g., 90-day periods)
        # TODO: Reshape accordingly
        
        # Option 1: 4 seasons of ~91 days each (364 days total)
        truncated_data = data[:, :364]
        seasonal_data = truncated_data.view(num_cities, 4, 91)
        
        # Option 2: More realistic seasonal boundaries
        # TODO: Handle actual seasonal dates
        
        print(f"Seasonal shape: {seasonal_data.shape}")
        return seasonal_data
    
    # Test all reshaping methods
    weekly = reshape_to_weeks(temperatures)
    monthly = reshape_to_months(temperatures)
    seasonal = seasonal_analysis(temperatures)
    
    # TODO: Calculate seasonal statistics
    # TODO: Compare temperature patterns across seasons
```

**Problem 2: Natural Language Processing Data**
```python
def nlp_data_reshaping():
    """
    Handle text data reshaping for different NLP tasks:
    
    Tasks:
    1. Batch variable-length sequences
    2. Create sliding windows for n-gram analysis
    3. Reshape for different attention patterns
    """
    
    def pad_sequences(sequences, max_length=None, pad_value=0):
        """Pad variable length sequences to same length"""
        # Input: List of tensors with different lengths
        # Output: (batch_size, max_seq_length) with padding
        
        # TODO: Find maximum length or use provided max_length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        batch_size = len(sequences)
        padded_sequences = torch.full((batch_size, max_length), pad_value)
        attention_mask = torch.zeros(batch_size, max_length)
        
        # TODO: Pad sequences with zeros or special tokens
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            if seq_len > max_length:
                # Truncate if too long
                padded_sequences[i] = seq[:max_length]
                attention_mask[i] = 1
            else:
                # Pad if too short
                padded_sequences[i, :seq_len] = seq
                attention_mask[i, :seq_len] = 1
        
        # TODO: Create attention mask for padded positions
        return padded_sequences, attention_mask
    
    def create_ngrams(sequence, n=3):
        """Create n-grams from sequence using sliding window"""
        # Input: (sequence_length,)
        # Output: (num_ngrams, n)
        
        # TODO: Use unfold() or manual indexing
        if len(sequence) < n:
            # Handle edge case: sequence shorter than n
            return torch.empty(0, n, dtype=sequence.dtype)
        
        # Method 1: Using unfold
        ngrams = sequence.unfold(0, n, 1)
        
        # Method 2: Manual approach
        # ngrams = torch.stack([sequence[i:i+n] for i in range(len(sequence)-n+1)])
        
        print(f"Sequence length: {len(sequence)}")
        print(f"N-grams shape: {ngrams.shape}")
        return ngrams
    
    def prepare_for_transformer(batch_sequences, cls_token=101, sep_token=102):
        """Prepare sequences for transformer input"""
        # TODO: Add special tokens (CLS, SEP)
        # TODO: Create position encodings
        # TODO: Handle attention masks
        
        processed_sequences = []
        for seq in batch_sequences:
            # Add CLS token at beginning, SEP token at end
            processed_seq = torch.cat([
                torch.tensor([cls_token]),  # CLS token
                seq,                        # Original sequence
                torch.tensor([sep_token])   # SEP token
            ])
            processed_sequences.append(processed_seq)
        
        # Pad all sequences
        padded_seqs, attention_masks = pad_sequences(processed_sequences)
        
        # TODO: Create position encodings
        batch_size, seq_length = padded_seqs.shape
        position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
        
        return {
            'input_ids': padded_seqs,
            'attention_mask': attention_masks,
            'position_ids': position_ids
        }
    
    # Test with sample sequences
    sample_sequences = [
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([6, 7, 8]),
        torch.tensor([9, 10, 11, 12]),
        torch.tensor([13, 14]),
        torch.tensor([15, 16, 17, 18, 19, 20, 21])
    ]
    
    # Test padding
    padded, masks = pad_sequences(sample_sequences)
    print(f"Padded shape: {padded.shape}")
    print(f"Attention mask shape: {masks.shape}")
    
    # Test n-grams
    test_sequence = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    trigrams = create_ngrams(test_sequence, n=3)
    print(f"Trigrams: {trigrams}")
    
    # Test transformer preparation
    transformer_inputs = prepare_for_transformer(sample_sequences[:3])
    for key, value in transformer_inputs.items():
        print(f"{key} shape: {value.shape}")

def broadcasting_challenges():
    """
    Implement advanced operations using broadcasting:
    """
    
    def batch_bias_addition():
        """Add bias vector to each sample in batch"""
        # Input: features (batch_size, num_features), bias (num_features,)
        # Output: features + bias for each sample
        
        batch_features = torch.randn(32, 128)  # 32 samples, 128 features each
        bias_vector = torch.randn(128)         # Bias for each feature
        
        # Broadcasting: (32, 128) + (128,) -> (32, 128)
        result = batch_features + bias_vector
        
        print(f"Features shape: {batch_features.shape}")
        print(f"Bias shape: {bias_vector.shape}")
        print(f"Result shape: {result.shape}")
        return result
    
    def per_sample_scaling():
        """Scale each sample differently"""
        # Input: features (batch_size, num_features), scales (batch_size, 1)
        
        batch_features = torch.randn(32, 128)
        sample_scales = torch.randn(32, 1)  # Different scale for each sample
        
        # Broadcasting: (32, 128) * (32, 1) -> (32, 128)
        result = batch_features * sample_scales
        
        print(f"Features shape: {batch_features.shape}")
        print(f"Scales shape: {sample_scales.shape}")
        print(f"Result shape: {result.shape}")
        return result
    
    def pairwise_distances():
        """Compute pairwise distances between points"""
        # Input: points (n_points, dimensions)
        # Output: distance matrix (n_points, n_points)
        
        points = torch.randn(20, 3)  # 20 points in 3D space
        
        # Method using broadcasting
        # |a - b|² = |a|² + |b|² - 2a·b
        
        # TODO: Compute squared norms for each point
        squared_norms = torch.sum(points ** 2, dim=1, keepdim=True)  # (20, 1)
        
        # TODO: Compute dot products between all pairs
        dot_products = torch.matmul(points, points.t())  # (20, 20)
        
        # TODO: Use broadcasting to compute distances
        # squared_norms: (20, 1), squared_norms.t(): (1, 20)
        # Broadcasting: (20, 1) + (1, 20) - 2*(20, 20) -> (20, 20)
        distances_squared = squared_norms + squared_norms.t() - 2 * dot_products
        distances = torch.sqrt(torch.clamp(distances_squared, min=0))
        
        print(f"Points shape: {points.shape}")
        print(f"Distance matrix shape: {distances.shape}")
        return distances
    
    def simplified_attention():
        """Implement simplified attention mechanism"""
        # Input: Query (batch, seq_len, d_model), Key (batch, seq_len, d_model)
        # Output: Attention weights (batch, seq_len, seq_len)
        
        batch_size, seq_len, d_model = 4, 10, 64
        
        Q = torch.randn(batch_size, seq_len, d_model)  # Query
        K = torch.randn(batch_size, seq_len, d_model)  # Key
        V = torch.randn(batch_size, seq_len, d_model)  # Value
        
        # TODO: Compute attention scores: Q @ K^T
        # Use torch.matmul for batch matrix multiplication
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
        
        # TODO: Scale by sqrt(d_model)
        scaled_scores = attention_scores / torch.sqrt(torch.tensor(d_model, dtype=torch.float))
        
        # TODO: Apply softmax across correct dimension
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # TODO: Compute weighted values
        attended_values = torch.matmul(attention_weights, V)  # (batch, seq_len, d_model)
        
        print(f"Q shape: {Q.shape}")
        print(f"K shape: {K.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Attended values shape: {attended_values.shape}")
        
        return attention_weights, attended_values
    
    # Test all broadcasting operations
    print("=== Testing Broadcasting Operations ===")
    batch_bias_addition()
    per_sample_scaling()
    pairwise_distances()
    simplified_attention()
```

#### Evaluation Criteria
- [ ] Correctly handles all shape transformations
- [ ] Preserves data integrity during reshaping
- [ ] Handles edge cases appropriately
- [ ] Code is efficient and readable

---

## Week 3: Autograd Fundamentals

### Learning Objectives
- Memorize autograd syntax patterns
- Understand requires_grad and gradient computation
- Apply automatic differentiation to solve optimization problems

---

### Exercise 3.1: Autograd Syntax Drill
**Type**: Muscle Memory  
**Time**: 3-4 hours  
**Difficulty**: Beginner

#### Objective
Memorize autograd operations through repetitive coding.

#### Requirements
Practice each autograd pattern **EXACTLY 20 times** following these specific requirements:

**TEMPLATE PATTERNS TO FOLLOW:**

```python
import torch

# Template for Creating tensors with gradients:
x = torch.tensor(value, requires_grad=True)  # comment describing tensor type
# or
x = torch.randn(shape)
x.requires_grad_(True)  # comment describing in-place setting

# Template for Basic forward and backward:
x = torch.tensor(value, requires_grad=True)  # input
y = function_of_x  # forward computation
y.backward()       # compute gradients  
print(f"x.grad: {x.grad}")  # access gradient
x.grad.zero_()     # reset gradients

# Template for Multi-variable functions:
x = torch.tensor(value1, requires_grad=True)
y = torch.tensor(value2, requires_grad=True)
z = function_of_x_and_y  # computation
z.backward()             # compute all gradients
print(f"x.grad: {x.grad}, y.grad: {y.grad}")
x.grad.zero_()  # reset
y.grad.zero_()  # reset

# Template for Vector operations:
x = torch.randn(shape, requires_grad=True)
y = vector_function(x).sum()  # sum to get scalar
y.backward()
print(f"x.grad: {x.grad}")
x.grad.zero_()
```

**Practice Set 1: Creating tensors with gradients - 20 REQUIRED variants**
Method 1: requires_grad parameter
- Variant 1: Create scalar tensor with value 2.0 and requires_grad=True
- Variant 2: Create 1D tensor [1.0, 2.0, 3.0] with requires_grad=True  
- Variant 3: Create 2D random tensor (3,4) with requires_grad=True
- Variant 4: Create scalar tensor with value 5.0 and requires_grad=True
- Variant 5: Create 1D tensor [0.5, 1.5, 2.5] with requires_grad=True
- Variant 6: Create 2D random tensor (2,5) with requires_grad=True
- Variant 7: Create scalar tensor with value -1.0 and requires_grad=True
- Variant 8: Create 1D tensor with 10 elements and requires_grad=True
- Variant 9: Create 3D random tensor (2,3,4) with requires_grad=True
- Variant 10: Create scalar tensor with value 0.0 and requires_grad=True

Method 2: requires_grad_() method
- Variant 11: Create random tensor (3,4), then set requires_grad_(True)
- Variant 12: Create zeros tensor (5,5), then set requires_grad_(True)
- Variant 13: Create ones tensor (2,3), then set requires_grad_(True)
- Variant 14: Create random tensor (10,), then set requires_grad_(True)
- Variant 15: Create scalar tensor 3.0, then set requires_grad_(True)
- Variant 16: Create arange tensor 0-9, then set requires_grad_(True)
- Variant 17: Create random tensor (1,1), then set requires_grad_(True)
- Variant 18: Create linspace tensor 0-1 with 11 points, then set requires_grad_(True)
- Variant 19: Create eye tensor (3,3), then set requires_grad_(True)
- Variant 20: Create full tensor (2,2) value=5.0, then set requires_grad_(True)

**Practice Set 2: Basic forward and backward - 20 REQUIRED variants**
Function: y = x^2 (with different x values)
- Variant 1: x = 3.0, compute y = x^2, backward, check grad = 2*x = 6
- Variant 2: x = 2.0, compute y = x^2, backward, check grad = 2*x = 4  
- Variant 3: x = -1.0, compute y = x^2, backward, check grad = 2*x = -2
- Variant 4: x = 0.5, compute y = x^2, backward, check grad = 2*x = 1
- Variant 5: x = 10.0, compute y = x^2, backward, check grad = 2*x = 20
- Variant 6: x = -5.0, compute y = x^2, backward, check grad = 2*x = -10
- Variant 7: x = 0.1, compute y = x^2, backward, check grad = 2*x = 0.2
- Variant 8: x = 7.0, compute y = x^2, backward, check grad = 2*x = 14
- Variant 9: x = -0.5, compute y = x^2, backward, check grad = 2*x = -1
- Variant 10: x = 1.5, compute y = x^2, backward, check grad = 2*x = 3

Other functions:
- Variant 11: x = 2.0, compute y = x^3, backward, check grad = 3*x^2 = 12
- Variant 12: x = 3.0, compute y = 2*x + 1, backward, check grad = 2
- Variant 13: x = 1.0, compute y = x^4, backward, check grad = 4*x^3 = 4
- Variant 14: x = -2.0, compute y = x^3, backward, check grad = 3*x^2 = 12
- Variant 15: x = 4.0, compute y = torch.sin(x), backward, check grad = cos(x)
- Variant 16: x = 1.0, compute y = torch.exp(x), backward, check grad = exp(x)
- Variant 17: x = 2.0, compute y = torch.log(x), backward, check grad = 1/x = 0.5
- Variant 18: x = 0.5, compute y = 1/x, backward, check grad = -1/x^2 = -4
- Variant 19: x = 3.0, compute y = torch.sqrt(x), backward, check grad = 1/(2*sqrt(x))
- Variant 20: x = 1.0, compute y = x^2 + 3*x + 2, backward, check grad = 2*x + 3 = 5

**Practice Set 3: Multi-variable functions - 20 REQUIRED variants**
Function: z = x^2 + y^2
- Variant 1: x=2.0, y=3.0, check x.grad=4, y.grad=6
- Variant 2: x=1.0, y=1.0, check x.grad=2, y.grad=2
- Variant 3: x=-1.0, y=2.0, check x.grad=-2, y.grad=4
- Variant 4: x=0.5, y=1.5, check x.grad=1, y.grad=3
- Variant 5: x=3.0, y=-2.0, check x.grad=6, y.grad=-4

Other multi-variable functions:
- Variant 6: z = x*y, with x=2.0, y=3.0, check x.grad=3, y.grad=2
- Variant 7: z = x^2*y, with x=2.0, y=3.0, check x.grad=2*x*y=12, y.grad=x^2=4
- Variant 8: z = x + y^2, with x=1.0, y=2.0, check x.grad=1, y.grad=4
- Variant 9: z = x^3 + y^3, with x=1.0, y=1.0, check x.grad=3, y.grad=3
- Variant 10: z = x*y^2, with x=2.0, y=3.0, check x.grad=9, y.grad=12
- Variant 11: z = torch.sin(x) + torch.cos(y), with x=0.0, y=0.0
- Variant 12: z = x^2 - y^2, with x=3.0, y=2.0, check x.grad=6, y.grad=-4
- Variant 13: z = x/y, with x=6.0, y=2.0, check x.grad=0.5, y.grad=-1.5
- Variant 14: z = x^2 + 2*x*y + y^2, with x=1.0, y=1.0
- Variant 15: z = torch.exp(x + y), with x=0.0, y=0.0
- Variant 16: z = x^2*y + x*y^2, with x=2.0, y=1.0
- Variant 17: z = torch.sqrt(x^2 + y^2), with x=3.0, y=4.0
- Variant 18: z = x^3*y^2, with x=1.0, y=2.0
- Variant 19: z = torch.log(x*y), with x=2.0, y=3.0
- Variant 20: z = (x + y)^2, with x=1.0, y=2.0

**Practice Set 4: Vector operations - 20 REQUIRED variants**
- Variant 1: x = random(3), y = (x^2).sum(), check grad = 2*x
- Variant 2: x = random(5), y = x.sum(), check grad = ones(5)
- Variant 3: x = random(4), y = (x^3).sum(), check grad = 3*x^2
- Variant 4: x = random(2), y = (x**4).sum(), check grad = 4*x^3
- Variant 5: x = random(6), y = torch.norm(x), check grad = x/norm(x)
- Variant 6: x = random(3), y = (torch.sin(x)).sum(), check grad = cos(x)
- Variant 7: x = random(4), y = (torch.exp(x)).sum(), check grad = exp(x)
- Variant 8: x = random(2,3), y = x.sum(), check grad = ones(2,3)
- Variant 9: x = random(3), y = torch.dot(x, x), check grad = 2*x
- Variant 10: x = random(5), y = (x * 2).sum(), check grad = 2*ones(5)
- Variant 11: x = random(2,2), y = (x**2).sum(), check grad = 2*x
- Variant 12: x = random(4), y = (1/x).sum(), check grad = -1/x^2
- Variant 13: x = random(3), y = torch.mean(x**2), check grad = 2*x/3
- Variant 14: x = random(2,3), y = torch.trace(x @ x.T), check grad pattern
- Variant 15: x = random(5), y = torch.var(x), check variance gradient
- Variant 16: x = random(3), y = torch.prod(x), check product gradient
- Variant 17: x = random(4), y = torch.max(x), check max gradient
- Variant 18: x = random(2,2), y = torch.det(x), check determinant gradient  
- Variant 19: x = random(3), y = torch.softmax(x, dim=0).sum(), check softmax grad
- Variant 20: x = random(5), y = torch.cumsum(x, dim=0).sum(), check cumsum grad

**Practice Set 5: Common gradient patterns - 20 REQUIRED variants**

**Template pattern:**
```python
def practice_pattern():
    # Setup tensors with requires_grad=True
    # Define computation
    # Compute loss (scalar)
    # Call loss.backward()
    # Print gradient shapes and verify expected values
```

**Linear Layer Patterns (10 variants):**
- Variant 1: x(5), W(3,5), b(3) → y = W@x + b, loss = y.sum()
- Variant 2: x(10), W(1,10), b(1) → linear regression pattern
- Variant 3: x(784), W(128,784), b(128) → MNIST-like input layer
- Variant 4: x(256), W(10,256), b(10) → classification output layer
- Variant 5: x(100), W(50,100), b(50) → dimension reduction layer
- Variant 6: x(20), W(64,20), b(64) → expansion layer
- Variant 7: Multi-layer: x(10) → W1(20,10) → W2(5,20) → output
- Variant 8: Batch processing: x(32,784), W(256,784), b(256)
- Variant 9: No bias: x(15), W(8,15), y = W@x (no bias term)
- Variant 10: Different activation: x(12), W(6,12), y = torch.relu(W@x)

**Matrix Operations Patterns (5 variants):**
- Variant 11: Matrix multiplication: A(3,4), B(4,5) → C = A@B, loss = C.sum()
- Variant 12: Quadratic form: x(5), A(5,5) → y = x.T @ A @ x (scalar output)
- Variant 13: Outer product: x(3), y(4) → Z = x.unsqueeze(1) @ y.unsqueeze(0)
- Variant 14: Trace operation: A(4,4) → loss = torch.trace(A @ A.T)
- Variant 15: Frobenius norm: A(3,4) → loss = torch.norm(A, 'fro')

**CNN-like Patterns (3 variants):**
- Variant 16: 1D convolution: x(1,10,20), weight(5,10,3) → conv1d operation
- Variant 17: 2D convolution: x(1,3,32,32), weight(16,3,3,3) → conv2d operation  
- Variant 18: Pooling gradients: x(1,1,4,4) → maxpool2d(2) → gradients

**Complex Patterns (2 variants):**
- Variant 19: RNN-like: x(seq_len,batch,input_size), hidden states, recurrent computation
- Variant 20: Attention-like: Q(10,64), K(15,64), V(15,64) → attention mechanism gradients

**Requirements for each variant:**
1. **Print input shapes**: Show shapes of all tensors with requires_grad=True
2. **Print gradient shapes**: Verify gradients match input shapes
3. **Verify gradient values**: Check gradients make mathematical sense
4. **Different dimensions**: Use varied tensor sizes to understand patterns
5. **Error handling**: Try operations that might fail, understand why

**Example implementation for Variant 1:**
```python
def practice_linear_grad_v1():
    print("=== Variant 1: Basic Linear Layer ===")
    x = torch.randn(5, requires_grad=True)
    W = torch.randn(3, 5, requires_grad=True)  
    b = torch.randn(3, requires_grad=True)
    
    print(f"Input shapes: x{x.shape}, W{W.shape}, b{b.shape}")
    
    y = torch.matmul(W, x) + b  # Linear transformation
    loss = y.sum()              # Scalar loss
    loss.backward()             # Compute gradients
    
    print(f"Gradient shapes: x.grad{x.grad.shape}, W.grad{W.grad.shape}, b.grad{b.grad.shape}")
    print(f"Loss value: {loss.item():.4f}")
    print(f"x.grad: {x.grad}")
    print(f"W.grad: {W.grad}")  
    print(f"b.grad: {b.grad}")
    print()

# Call this function once, then implement 19 more variants following the pattern
practice_linear_grad_v1()
```

**CRITICAL**: You must implement ALL 20 variants. Each variant teaches different gradient patterns!
```

#### Deliverables
- **autograd_drill.py**: 100+ autograd operations
- **gradient_patterns.py**: Common gradient computation patterns

---

### Exercise 3.2: requires_grad Context Practice
**Type**: Understanding + Repetition  
**Time**: 2-3 hours  
**Difficulty**: Beginner

#### Objective
Master when and how to use requires_grad through repeated practice.

#### Requirements
Practice requires_grad in different contexts 30 times:

```python
import torch

# Practice Set 1: Training vs Inference modes (repeat 15 times)
def training_mode_practice():
    # Training: need gradients
    x = torch.randn(10, requires_grad=True)
    W = torch.randn(5, 10, requires_grad=True)
    
    # Forward pass
    y = torch.matmul(W, x)
    loss = y.sum()
    
    # Backward pass
    loss.backward()
    
    print("Training mode - gradients computed")
    print(f"W.grad is not None: {W.grad is not None}")

def inference_mode_practice():
    # Inference: no gradients needed
    with torch.no_grad():
        x = torch.randn(10)  # No requires_grad needed
        W = torch.randn(5, 10)  # No requires_grad needed
        
        y = torch.matmul(W, x)
        loss = y.sum()
        
        print("Inference mode - no gradients")
        print(f"x.requires_grad: {x.requires_grad}")

# Practice both modes 15 times each
for i in range(15):
    training_mode_practice()
    inference_mode_practice()

# Practice Set 2: Detaching tensors (repeat 15 times)
def detach_practice():
    x = torch.randn(5, requires_grad=True)
    y = x**2
    
    # Detach y from computation graph
    y_detached = y.detach()
    
    # Further computation with detached tensor
    z = y_detached * 2
    
    print(f"x.requires_grad: {x.requires_grad}")
    print(f"y.requires_grad: {y.requires_grad}")
    print(f"y_detached.requires_grad: {y_detached.requires_grad}")
    print(f"z.requires_grad: {z.requires_grad}")

for i in range(15):
    detach_practice()
```

#### Daily Practice Routine
- **Day 1-3**: Basic requires_grad patterns
- **Day 4-5**: Training vs inference modes
- **Day 6-7**: Detaching and context managers

---

### Exercise 3.3: Optimization Applications
**Type**: Thinking + Application  
**Time**: 4-5 hours  
**Difficulty**: Intermediate

#### Objective
Use autograd to solve calculus and optimization problems.

#### Requirements

**Problem 1: Function Analysis**
```python
def function_analyzer():
    """
    Analyze mathematical functions using autograd:
    1. Find critical points (where gradient = 0)
    2. Determine if critical points are minima/maxima
    3. Plot function and its derivative
    """
    
    def analyze_polynomial(coefficients):
        """
        Analyze polynomial f(x) = a₃x³ + a₂x² + a₁x + a₀
        Find critical points and classify them.
        """
        # TODO: Define polynomial function
        # TODO: Use autograd to find derivative
        # TODO: Solve f'(x) = 0 numerically
        # TODO: Use second derivative test for classification
        
        def polynomial(x, coeffs):
            """Evaluate polynomial at x"""
            # TODO: Implement polynomial evaluation
            result = torch.zeros_like(x)
            for i, coeff in enumerate(coeffs):
                power = len(coeffs) - 1 - i
                result += coeff * (x ** power)
            return result
        
        def find_critical_points(coeffs, x_range=(-10, 10), num_points=1000):
            """Find critical points in given range"""
            # TODO: Sample points in range
            x_vals = torch.linspace(x_range[0], x_range[1], num_points, requires_grad=True)
            
            # TODO: Compute gradients
            y_vals = polynomial(x_vals, coeffs)
            
            # Compute gradient for each point
            gradients = []
            for i in range(len(x_vals)):
                if x_vals.grad is not None:
                    x_vals.grad.zero_()
                
                y_vals[i].backward(retain_graph=True)
                gradients.append(x_vals.grad[i].item())
            
            # TODO: Find where gradient ≈ 0
            gradients = torch.tensor(gradients)
            near_zero_indices = torch.abs(gradients) < 0.01
            critical_x = x_vals[near_zero_indices].detach()
            
            return critical_x
        
        # Test with polynomial: x³ - 6x² + 9x + 1
        test_coeffs = torch.tensor([1.0, -6.0, 9.0, 1.0])
        critical_points = find_critical_points(test_coeffs)
        
        print(f"Critical points found: {critical_points}")
        return critical_points
    
    def optimization_surface():
        """
        Analyze 2D optimization surface: f(x,y) = x² + y² - 2xy + 3x - 4y + 5
        Find minimum using gradient descent with autograd.
        """
        
        def objective_function(x, y):
            return x**2 + y**2 - 2*x*y + 3*x - 4*y + 5
        
        def gradient_descent_step(x, y, learning_rate=0.01):
            """Single step of gradient descent"""
            # TODO: Compute gradients using autograd
            # TODO: Update x, y using gradients
            # TODO: Return new x, y and function value
            
            # Clear previous gradients
            if x.grad is not None:
                x.grad.zero_()
            if y.grad is not None:
                y.grad.zero_()
            
            # Forward pass
            loss = objective_function(x, y)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            with torch.no_grad():
                x -= learning_rate * x.grad
                y -= learning_rate * y.grad
            
            return x, y, loss.item()
        
        def find_minimum(start_x=0.0, start_y=0.0, max_iterations=1000, tolerance=1e-6):
            """Find minimum using gradient descent"""
            # TODO: Initialize variables with requires_grad=True
            x = torch.tensor(start_x, requires_grad=True)
            y = torch.tensor(start_y, requires_grad=True)
            
            history = {'x': [], 'y': [], 'loss': []}
            
            # TODO: Iterate gradient descent steps
            for i in range(max_iterations):
                x, y, loss = gradient_descent_step(x, y)
                
                history['x'].append(x.item())
                history['y'].append(y.item())
                history['loss'].append(loss)
                
                # TODO: Track convergence
                if i > 0 and abs(history['loss'][-2] - history['loss'][-1]) < tolerance:
                    print(f"Converged after {i+1} iterations")
                    break
            
            # TODO: Return optimal point and value
            print(f"Minimum found at: x={x.item():.6f}, y={y.item():.6f}")
            print(f"Minimum value: {loss:.6f}")
            
            return x.item(), y.item(), loss, history
        
        # Find minimum
        optimal_x, optimal_y, min_value, history = find_minimum()
        
        # Compare with analytical solution
        # For f(x,y) = x² + y² - 2xy + 3x - 4y + 5
        # Analytical minimum is at x = 0.5, y = 2.5
        print(f"Analytical solution: x=0.5, y=2.5")
        print(f"Difference: dx={abs(optimal_x - 0.5):.6f}, dy={abs(optimal_y - 2.5):.6f}")
        
        return history
    
    # Run both analyses
    analyze_polynomial([1.0, -6.0, 9.0, 1.0])
    optimization_surface()

def physics_with_autograd():
    """
    Use autograd for physics calculations:
    1. Projectile motion optimization
    2. Energy function analysis
    3. Force field calculations
    """
    
    def projectile_optimization():
        """
        Find optimal launch angle for maximum range:
        - Given initial velocity v₀
        - Find angle θ that maximizes range
        - Use autograd to optimize
        """
        
        def range_function(angle, v0=20.0, g=9.81):
            """Calculate projectile range for given angle"""
            # TODO: Implement range formula: R = v₀²sin(2θ)/g
            # TODO: Use torch.sin() for autograd compatibility
            range_val = (v0**2 * torch.sin(2 * angle)) / g
            return range_val
        
        def find_optimal_angle():
            """Use gradient ascent to find optimal angle"""
            # TODO: Initialize angle with requires_grad=True
            angle = torch.tensor(0.5, requires_grad=True)  # Start at ~30 degrees
            
            optimizer = torch.optim.Adam([angle], lr=0.01)
            
            # TODO: Maximize range using gradient ascent
            for i in range(1000):
                optimizer.zero_grad()
                
                # We want to maximize range, so minimize -range
                loss = -range_function(angle)
                loss.backward()
                optimizer.step()
                
                # Keep angle in valid range [0, π/2]
                with torch.no_grad():
                    angle.clamp_(0, torch.pi/2)
                
                if i % 100 == 0:
                    current_range = range_function(angle).item()
                    print(f"Iteration {i}: angle={angle.item():.4f} rad ({torch.rad2deg(angle).item():.2f}°), range={current_range:.2f}m")
            
            # TODO: Compare with analytical solution (45°)
            optimal_angle_deg = torch.rad2deg(angle).item()
            optimal_range = range_function(angle).item()
            
            print(f"\nOptimal angle found: {optimal_angle_deg:.2f}°")
            print(f"Maximum range: {optimal_range:.2f}m")
            print(f"Analytical solution: 45°")
            print(f"Error: {abs(optimal_angle_deg - 45):.2f}°")
            
            return angle.item(), optimal_range
        
        return find_optimal_angle()
    
    def spring_system():
        """
        Analyze spring-mass system energy:
        - Total energy E = ½kx² + ½mv² (kinetic + potential)
        - Find equilibrium position
        - Analyze stability
        """
        
        def total_energy(position, velocity, k=1.0, m=1.0):
            """Calculate total energy of spring-mass system"""
            # TODO: Implement energy function
            kinetic_energy = 0.5 * m * velocity**2
            potential_energy = 0.5 * k * position**2
            return kinetic_energy + potential_energy
        
        def find_equilibrium():
            """Find equilibrium using energy minimization"""
            # TODO: Minimize potential energy
            position = torch.tensor(1.0, requires_grad=True)  # Start away from equilibrium
            
            optimizer = torch.optim.Adam([position], lr=0.1)
            
            for i in range(100):
                optimizer.zero_grad()
                
                # At equilibrium, only potential energy matters (velocity = 0)
                potential_energy = 0.5 * position**2
                potential_energy.backward()
                optimizer.step()
                
                if i % 20 == 0:
                    print(f"Iteration {i}: position={position.item():.6f}, potential_energy={potential_energy.item():.6f}")
            
            # TODO: Verify with analytical solution
            print(f"\nEquilibrium position found: {position.item():.6f}")
            print(f"Analytical solution: 0.0")
            print(f"Error: {abs(position.item()):.6f}")
            
            return position.item()
        
        return find_equilibrium()
    
    # Run physics simulations
    print("=== Projectile Motion Optimization ===")
    projectile_optimization()
    
    print("\n=== Spring System Analysis ===")
    spring_system()

def ml_optimization():
    """
    Implement basic ML algorithms using autograd:
    1. Linear regression from scratch
    2. Logistic regression with gradient descent  
    3. Simple neural network optimization
    """
    
    def linear_regression_autograd():
        """
        Implement linear regression using only autograd (no nn.Module):
        - Manually define parameters w, b
        - Implement MSE loss
        - Use gradient descent for optimization
        """
        
        # Generate sample data
        torch.manual_seed(42)
        n_samples = 100
        X = torch.randn(n_samples, 1)
        true_w, true_b = 3.0, -2.0
        y = true_w * X + true_b + 0.1 * torch.randn(n_samples, 1)
        
        # TODO: Initialize parameters with requires_grad=True
        w = torch.randn(1, 1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)
        
        # Training loop
        learning_rate = 0.01
        num_epochs = 1000
        
        for epoch in range(num_epochs):
            # TODO: Implement forward pass manually
            y_pred = torch.matmul(X, w) + b
            
            # TODO: Implement MSE loss
            loss = torch.mean((y_pred - y)**2)
            
            # TODO: Use autograd for optimization loop
            if w.grad is not None:
                w.grad.zero_()
            if b.grad is not None:
                b.grad.zero_()
            
            loss.backward()
            
            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, w={w.item():.3f}, b={b.item():.3f}")
        
        # TODO: Compare learned parameters with true values
        print(f"\nFinal results:")
        print(f"Learned: w={w.item():.3f}, b={b.item():.3f}")
        print(f"True: w={true_w:.3f}, b={true_b:.3f}")
        print(f"Error: w_error={abs(w.item() - true_w):.3f}, b_error={abs(b.item() - true_b):.3f}")
        
        return w.item(), b.item()
    
    def logistic_regression_autograd():
        """
        Implement binary classification using autograd:
        - Manual sigmoid implementation
        - Binary cross-entropy loss
        - Gradient-based optimization
        """
        
        # Generate sample data
        torch.manual_seed(42)
        n_samples = 200
        X = torch.randn(n_samples, 2)
        true_w = torch.tensor([[1.5], [-2.0]])
        
        # Create decision boundary
        linear_combination = torch.matmul(X, true_w)
        y = (linear_combination > 0).float()
        
        # TODO: Initialize parameters
        w = torch.randn(2, 1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)
        
        def sigmoid(x):
            """Manual sigmoid implementation"""
            return 1 / (1 + torch.exp(-x))
        
        def binary_cross_entropy(y_pred, y_true):
            """Manual binary cross-entropy implementation"""
            # Clamp predictions to avoid log(0)
            y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
            return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        # Training loop
        learning_rate = 0.1
        num_epochs = 1000
        
        for epoch in range(num_epochs):
            # TODO: Implement forward pass
            linear_output = torch.matmul(X, w) + b
            y_pred = sigmoid(linear_output)
            
            # TODO: Implement binary cross-entropy loss
            loss = binary_cross_entropy(y_pred, y)
            
            # TODO: Optimize using autograd
            if w.grad is not None:
                w.grad.zero_()
            if b.grad is not None:
                b.grad.zero_()
            
            loss.backward()
            
            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad
            
            if epoch % 100 == 0:
                # Calculate accuracy
                predictions = (y_pred > 0.5).float()
                accuracy = torch.mean((predictions == y).float()).item()
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={accuracy:.3f}")
        
        # TODO: Evaluate classification accuracy
        with torch.no_grad():
            linear_output = torch.matmul(X, w) + b
            y_pred = sigmoid(linear_output)
            predictions = (y_pred > 0.5).float()
            final_accuracy = torch.mean((predictions == y).float()).item()
        
        print(f"\nFinal accuracy: {final_accuracy:.3f}")
        print(f"Learned weights: {w.squeeze().tolist()}")
        print(f"True weights: {true_w.squeeze().tolist()}")
        
        return w, b, final_accuracy
    
    def simple_neural_network():
        """
        Implement 2-layer neural network using only autograd
        """
        # Generate sample data for XOR problem
        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])  # XOR labels
        
        # Initialize parameters
        hidden_size = 10
        W1 = torch.randn(2, hidden_size, requires_grad=True) * 0.5
        b1 = torch.randn(hidden_size, requires_grad=True) * 0.5
        W2 = torch.randn(hidden_size, 1, requires_grad=True) * 0.5
        b2 = torch.randn(1, requires_grad=True) * 0.5
        
        def relu(x):
            return torch.clamp(x, min=0)
        
        def sigmoid(x):
            return 1 / (1 + torch.exp(-x))
        
        # Training loop
        learning_rate = 0.1
        num_epochs = 5000
        
        for epoch in range(num_epochs):
            # Forward pass
            hidden = relu(torch.matmul(X, W1) + b1)
            output = sigmoid(torch.matmul(hidden, W2) + b2)
            
            # Loss
            loss = torch.mean((output - y)**2)
            
            # Backward pass
            for param in [W1, b1, W2, b2]:
                if param.grad is not None:
                    param.grad.zero_()
            
            loss.backward()
            
            # Update parameters
            with torch.no_grad():
                W1 -= learning_rate * W1.grad
                b1 -= learning_rate * b1.grad
                W2 -= learning_rate * W2.grad
                b2 -= learning_rate * b2.grad
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}")
        
        # Test final performance
        with torch.no_grad():
            hidden = relu(torch.matmul(X, W1) + b1)
            output = sigmoid(torch.matmul(hidden, W2) + b2)
            predictions = (output > 0.5).float()
            accuracy = torch.mean((predictions == y).float()).item()
        
        print(f"\nXOR Network Results:")
        print(f"Final accuracy: {accuracy:.3f}")
        print(f"Predictions vs True:")
        for i in range(len(X)):
            print(f"  Input: {X[i].tolist()}, Predicted: {output[i].item():.3f}, True: {y[i].item()}")
        
        return accuracy
    
    # Run all ML algorithms
    print("=== Linear Regression ===")
    linear_regression_autograd()
    
    print("\n=== Logistic Regression ===")
    logistic_regression_autograd()
    
    print("\n=== Simple Neural Network (XOR) ===")
    simple_neural_network()
```

#### Evaluation Criteria
- [ ] Correctly uses autograd for gradient computation
- [ ] Implements optimization algorithms properly
- [ ] Validates results against analytical solutions
- [ ] Handles numerical stability issues

---

## Week 4: Neural Network Modules (nn.Module)

### Learning Objectives
- Memorize nn.Module patterns and syntax
- Understand basic neural network construction
- Build practical neural networks for real applications

---

### Exercise 4.1: nn.Module Syntax Drill
**Type**: Muscle Memory  
**Time**: 4-5 hours  
**Difficulty**: Beginner-Intermediate

#### Objective
Memorize nn.Module creation patterns through repetitive coding.

#### Requirements
Create **EXACTLY 25 different nn.Module classes** following these specific requirements:

**TEMPLATE PATTERNS TO FOLLOW:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Template for Basic Linear Network:
class NetworkName(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NetworkName, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], output_size)
        # Add more layers as needed
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))  # activation
        x = self.layer2(x)              # output layer
        return x

# Template for Sequential Network:
class SequentialNetwork(nn.Module):
    def __init__(self):
        super(SequentialNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Template for CNN:
class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x

# Template for Custom Logic:
class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.branch1 = nn.Linear(input_size, hidden_size)
        self.branch2 = nn.Linear(input_size, hidden_size)
        self.combine = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        branch1_out = torch.relu(self.branch1(x))
        branch2_out = torch.sigmoid(self.branch2(x))
        combined = torch.cat([branch1_out, branch2_out], dim=1)
        output = self.combine(combined)
        return output
```

**Practice Set 1: Basic Linear Networks - 5 REQUIRED variants**
- Variant 1: SimpleLinear1 - input(784) → hidden(128) → output(10), ReLU activation
- Variant 2: SimpleLinear2 - input(100) → hidden(256) → hidden(64) → output(1), ReLU activations
- Variant 3: SimpleLinear3 - input(50) → hidden(32) → hidden(16) → output(5), LeakyReLU activations
- Variant 4: SimpleLinear4 - input(200) → hidden(512) → hidden(256) → hidden(128) → output(2), ReLU + Dropout(0.2)
- Variant 5: SimpleLinear5 - input(300) → hidden(64) → output(20), with bias=False for output layer

**Practice Set 2: Sequential Networks - 5 REQUIRED variants**
- Variant 6: SequentialModel1 - MNIST classifier: (784,256,128,10) with ReLU, Dropout(0.2)
- Variant 7: SequentialModel2 - Binary classifier: (50,32,16,1) with ReLU, Sigmoid output
- Variant 8: SequentialModel3 - Regression: (20,64,32,1) with ReLU, no final activation
- Variant 9: SequentialModel4 - Multi-class: (100,512,256,128,64,10) with ReLU, Dropout(0.3)
- Variant 10: SequentialModel5 - Deep network: (784,1024,512,256,128,64,32,10) with ReLU, BatchNorm1d

**Practice Set 3: CNN Networks - 5 REQUIRED variants**
- Variant 11: SimpleCNN1 - MNIST: Conv2d(1,32,3) → Pool → Conv2d(32,64,3) → Pool → FC(64*5*5,128) → FC(128,10)
- Variant 12: SimpleCNN2 - CIFAR: Conv2d(3,64,3) → Pool → Conv2d(64,128,3) → Pool → FC layers with Dropout
- Variant 13: SimpleCNN3 - Grayscale: Conv2d(1,16,5) → Conv2d(16,32,5) → Conv2d(32,64,3) → FC layers
- Variant 14: SimpleCNN4 - RGB with BatchNorm: Conv2d→BatchNorm2d→ReLU→Pool pattern repeated
- Variant 15: SimpleCNN5 - Deep CNN: 4 conv layers (3→32→64→128→256) with adaptive pooling

**Practice Set 4: Mixed Architecture Networks - 5 REQUIRED variants**
- Variant 16: MixedModel1 - CNN backbone + FC head with residual connection
- Variant 17: MixedModel2 - Multi-input: 2 branches merge → shared FC layers
- Variant 18: MixedModel3 - Encoder-Decoder style with bottleneck
- Variant 19: MixedModel4 - CNN feature extractor + LSTM classifier
- Variant 20: MixedModel5 - Attention mechanism with FC layers

**Practice Set 5: Custom Logic Networks - 5 REQUIRED variants**
- Variant 21: CustomLogic1 - Residual connections: x + F(x) pattern
- Variant 22: CustomLogic2 - Multi-branch with different activations, concat outputs
- Variant 23: CustomLogic3 - Skip connections giữa non-adjacent layers
- Variant 24: CustomLogic4 - Conditional computation based on input features
- Variant 25: CustomLogic5 - Self-attention mechanism trong FC network

**CRITICAL**: You must implement ALL 25 classes. Each class teaches different nn.Module patterns!

#### Deliverables
- **nn_module_drill.py**: 25 different nn.Module implementations
- **module_testing.py**: Test code for each module
- **module_patterns.md**: Notes on common patterns

---

### Exercise 4.2: Layer Types Memorization
**Type**: Muscle Memory  
**Time**: 3-4 hours  
**Difficulty**: Beginner

#### Objective
Memorize syntax for all common PyTorch layers.

#### Requirements
Write initialization code for each layer type **EXACTLY 10 times** following these specific requirements:

**TEMPLATE PATTERNS TO FOLLOW:**

```python
import torch.nn as nn

# Template for Linear Layers:
linear_layer = nn.Linear(in_features, out_features, bias=True)  # comment about purpose

# Template for Convolutional Layers:
conv2d_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)  # comment about image processing
conv1d_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)  # comment about sequence processing

# Template for Pooling Layers:
maxpool_layer = nn.MaxPool2d(kernel_size, stride=None, padding=0)  # comment about downsampling
avgpool_layer = nn.AvgPool2d(kernel_size, stride=None, padding=0)  # comment about downsampling

# Template for Normalization Layers:
batchnorm_layer = nn.BatchNorm2d(num_features)  # comment about normalization
layernorm_layer = nn.LayerNorm(normalized_shape)  # comment about normalization

# Template for Activation Layers:
activation_layer = nn.ReLU()  # comment about non-linearity

# Template for Regularization Layers:
dropout_layer = nn.Dropout(p=0.5)  # comment about regularization

# Template for Recurrent Layers:
rnn_layer = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)  # comment about sequence modeling
```

**Practice Set 1: Linear Layers - 10 REQUIRED variants**
- Variant 1: nn.Linear(784, 256) - MNIST input to hidden
- Variant 2: nn.Linear(256, 128) - hidden to hidden
- Variant 3: nn.Linear(128, 10) - hidden to output classes
- Variant 4: nn.Linear(100, 50) - general purpose reduction
- Variant 5: nn.Linear(50, 100) - expansion layer
- Variant 6: nn.Linear(512, 1024) - large expansion
- Variant 7: nn.Linear(1024, 1) - regression output
- Variant 8: nn.Linear(300, 300) - same size transformation
- Variant 9: nn.Linear(784, 10, bias=False) - no bias linear
- Variant 10: nn.Linear(2048, 512) - feature reduction

**Practice Set 2: Conv2d Layers - 10 REQUIRED variants**
- Variant 1: nn.Conv2d(3, 64, kernel_size=3) - RGB input, 3x3 filter
- Variant 2: nn.Conv2d(64, 128, kernel_size=3, stride=2) - downsampling conv
- Variant 3: nn.Conv2d(128, 256, kernel_size=3, padding=1) - same size conv
- Variant 4: nn.Conv2d(1, 32, kernel_size=5) - grayscale input, 5x5 filter
- Variant 5: nn.Conv2d(32, 64, kernel_size=1) - 1x1 pointwise conv
- Variant 6: nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) - standard conv
- Variant 7: nn.Conv2d(3, 16, kernel_size=7, padding=3) - large kernel
- Variant 8: nn.Conv2d(512, 256, kernel_size=3, groups=256) - depthwise conv
- Variant 9: nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=2) - dilated conv
- Variant 10: nn.Conv2d(128, 64, kernel_size=3, bias=False) - no bias conv

**Practice Set 3: Conv1d Layers - 10 REQUIRED variants**
- Variant 1: nn.Conv1d(100, 256, kernel_size=3) - sequence feature extraction
- Variant 2: nn.Conv1d(256, 512, kernel_size=5, padding=2) - larger context
- Variant 3: nn.Conv1d(512, 256, kernel_size=1) - pointwise conv 1D
- Variant 4: nn.Conv1d(50, 128, kernel_size=7, stride=2) - downsampling 1D
- Variant 5: nn.Conv1d(128, 64, kernel_size=3, dilation=2) - dilated 1D conv
- Variant 6: nn.Conv1d(1, 32, kernel_size=15) - single channel to multi
- Variant 7: nn.Conv1d(32, 64, kernel_size=9, padding=4) - symmetric padding
- Variant 8: nn.Conv1d(64, 128, kernel_size=5, groups=32) - grouped conv 1D
- Variant 9: nn.Conv1d(300, 600, kernel_size=3, stride=1) - expansion conv
- Variant 10: nn.Conv1d(200, 100, kernel_size=11, padding=5) - large kernel 1D

**Practice Set 4: Pooling Layers - 10 REQUIRED variants**
- Variant 1: nn.MaxPool2d(kernel_size=2) - standard 2x2 max pooling
- Variant 2: nn.MaxPool2d(kernel_size=3, stride=2) - overlapping max pool
- Variant 3: nn.AvgPool2d(kernel_size=2) - average pooling 2x2
- Variant 4: nn.AvgPool2d(kernel_size=4, stride=4) - non-overlapping avg pool
- Variant 5: nn.AdaptiveMaxPool2d((1, 1)) - global max pooling
- Variant 6: nn.AdaptiveAvgPool2d((1, 1)) - global average pooling
- Variant 7: nn.AdaptiveAvgPool2d((7, 7)) - fixed output size
- Variant 8: nn.MaxPool1d(kernel_size=3) - 1D max pooling
- Variant 9: nn.AvgPool1d(kernel_size=5, stride=2) - 1D avg pooling
- Variant 10: nn.AdaptiveMaxPool1d(1) - global 1D max pooling

**Practice Set 5: Normalization Layers - 10 REQUIRED variants**
- Variant 1: nn.BatchNorm1d(256) - 1D batch normalization
- Variant 2: nn.BatchNorm2d(64) - 2D batch normalization
- Variant 3: nn.LayerNorm(512) - layer normalization
- Variant 4: nn.GroupNorm(32, 256) - group normalization
- Variant 5: nn.InstanceNorm2d(128) - instance normalization
- Variant 6: nn.BatchNorm1d(1024, momentum=0.1) - custom momentum
- Variant 7: nn.LayerNorm([256, 256]) - 2D layer norm
- Variant 8: nn.GroupNorm(8, 128) - fewer groups
- Variant 9: nn.BatchNorm2d(32, eps=1e-3) - custom epsilon
- Variant 10: nn.LocalResponseNorm(5) - local response normalization

**Practice Set 6: Activation Layers - 10 REQUIRED variants**
- Variant 1: nn.ReLU() - standard ReLU
- Variant 2: nn.LeakyReLU(0.01) - leaky ReLU
- Variant 3: nn.Sigmoid() - sigmoid activation
- Variant 4: nn.Tanh() - hyperbolic tangent
- Variant 5: nn.GELU() - Gaussian Error Linear Unit
- Variant 6: nn.Softmax(dim=1) - softmax normalization
- Variant 7: nn.LogSoftmax(dim=1) - log softmax
- Variant 8: nn.ELU(alpha=1.0) - Exponential Linear Unit
- Variant 9: nn.Swish() - Swish activation
- Variant 10: nn.Mish() - Mish activation

**Practice Set 7: Regularization Layers - 10 REQUIRED variants**
- Variant 1: nn.Dropout(0.5) - standard dropout
- Variant 2: nn.Dropout(0.2) - light dropout
- Variant 3: nn.Dropout2d(0.1) - 2D dropout
- Variant 4: nn.Dropout(0.8) - heavy dropout
- Variant 5: nn.AlphaDropout(0.5) - SELU compatible dropout
- Variant 6: nn.Dropout2d(0.25) - 2D dropout for conv layers
- Variant 7: nn.Dropout(0.3) - moderate dropout
- Variant 8: nn.Dropout(p=0.0) - no dropout (placeholder)
- Variant 9: nn.Dropout3d(0.15) - 3D dropout
- Variant 10: nn.FeatureAlphaDropout(0.4) - feature alpha dropout

**Practice Set 8: Recurrent Layers - 10 REQUIRED variants**
- Variant 1: nn.LSTM(100, 256, batch_first=True) - basic LSTM
- Variant 2: nn.LSTM(256, 512, num_layers=2, batch_first=True) - deep LSTM
- Variant 3: nn.GRU(100, 256, batch_first=True) - basic GRU
- Variant 4: nn.RNN(100, 256, batch_first=True) - vanilla RNN
- Variant 5: nn.LSTM(512, 256, dropout=0.2, batch_first=True) - LSTM with dropout
- Variant 6: nn.GRU(256, 128, num_layers=3, batch_first=True) - deep GRU
- Variant 7: nn.LSTM(300, 600, bidirectional=True, batch_first=True) - bidirectional LSTM
- Variant 8: nn.GRU(200, 400, bidirectional=True, batch_first=True) - bidirectional GRU
- Variant 9: nn.RNN(150, 300, num_layers=2, nonlinearity='tanh', batch_first=True) - tanh RNN
- Variant 10: nn.LSTM(64, 128, proj_size=64, batch_first=True) - LSTM with projection

**Practice Set 9: Embedding Layers - 10 REQUIRED variants**
- Variant 1: nn.Embedding(10000, 300) - word embeddings
- Variant 2: nn.Embedding(5000, 128, padding_idx=0) - with padding
- Variant 3: nn.Embedding(50000, 512) - large vocabulary
- Variant 4: nn.Embedding(1000, 64, max_norm=1.0) - normalized embeddings
- Variant 5: nn.Embedding(20000, 256, sparse=True) - sparse embeddings
- Variant 6: nn.EmbeddingBag(10000, 300, mode='mean') - embedding bag
- Variant 7: nn.Embedding(2000, 100, scale_grad_by_freq=True) - scaled gradients
- Variant 8: nn.Embedding(15000, 768) - BERT-like embeddings
- Variant 9: nn.Embedding(30000, 200, padding_idx=1) - different padding
- Variant 10: nn.EmbeddingBag(5000, 150, mode='sum') - sum pooling embeddings

**CRITICAL**: You must write ALL 100 layer declarations (10 × 10 layer types). Memorize syntax of every layer!

#### Daily Memorization Schedule
- **Day 1**: Linear, Conv2d, MaxPool2d
- **Day 2**: BatchNorm, Dropout, ReLU
- **Day 3**: LSTM, GRU, Embedding
- **Day 4**: Review all layers
- **Day 5-7**: Speed drill - write all layers from memory

---

### Exercise 4.3: Architecture Design Applications
**Type**: Thinking + Application  
**Time**: 5-6 hours  
**Difficulty**: Intermediate

#### Objective
Design and implement neural networks for specific problem domains.

#### Requirements

**Challenge 1: Custom Layer Implementation**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearWithStats(nn.Module):
    """
    Linear layer that tracks and reports statistics:
    - Input/output statistics (mean, std, min, max)
    - Weight and gradient norms
    - Activation patterns
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithStats, self).__init__()
        
        # TODO: Initialize linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # TODO: Initialize statistics tracking variables
        self.register_buffer('input_mean_running', torch.zeros(1))
        self.register_buffer('input_std_running', torch.ones(1))
        self.register_buffer('output_mean_running', torch.zeros(1))
        self.register_buffer('output_std_running', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # TODO: Register buffers for running statistics
        self.momentum = 0.1
        self.input_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
        self.output_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
        self.weight_norms = []
        self.grad_norms = []
        
    def forward(self, x):
        # TODO: Compute linear transformation
        output = self.linear(x)
        
        # TODO: Update input statistics
        if self.training:
            with torch.no_grad():
                input_mean = x.mean()
                input_std = x.std()
                input_min = x.min()
                input_max = x.max()
                
                # Update running statistics
                if self.num_batches_tracked == 0:
                    self.input_mean_running = input_mean
                    self.input_std_running = input_std
                else:
                    self.input_mean_running = (1 - self.momentum) * self.input_mean_running + self.momentum * input_mean
                    self.input_std_running = (1 - self.momentum) * self.input_std_running + self.momentum * input_std
                
                # Store batch statistics
                self.input_stats['mean'].append(input_mean.item())
                self.input_stats['std'].append(input_std.item())
                self.input_stats['min'].append(input_min.item())
                self.input_stats['max'].append(input_max.item())
        
        # TODO: Update output statistics
        if self.training:
            with torch.no_grad():
                output_mean = output.mean()
                output_std = output.std()
                output_min = output.min()
                output_max = output.max()
                
                # Update running statistics
                if self.num_batches_tracked == 0:
                    self.output_mean_running = output_mean
                    self.output_std_running = output_std
                else:
                    self.output_mean_running = (1 - self.momentum) * self.output_mean_running + self.momentum * output_mean
                    self.output_std_running = (1 - self.momentum) * self.output_std_running + self.momentum * output_std
                
                # Store batch statistics
                self.output_stats['mean'].append(output_mean.item())
                self.output_stats['std'].append(output_std.item())
                self.output_stats['min'].append(output_min.item())
                self.output_stats['max'].append(output_max.item())
                
                # Track weight norms
                weight_norm = torch.norm(self.linear.weight)
                self.weight_norms.append(weight_norm.item())
                
                # Track gradient norms (if available)
                if self.linear.weight.grad is not None:
                    grad_norm = torch.norm(self.linear.weight.grad)
                    self.grad_norms.append(grad_norm.item())
                
                self.num_batches_tracked += 1
        
        # TODO: Return output
        return output
    
    def get_statistics(self):
        """Return current statistics as dictionary"""
        # TODO: Return comprehensive statistics
        stats = {
            'input_stats': {
                'running_mean': self.input_mean_running.item(),
                'running_std': self.input_std_running.item(),
                'batch_means': self.input_stats['mean'][-10:],  # Last 10 batches
                'batch_stds': self.input_stats['std'][-10:]
            },
            'output_stats': {
                'running_mean': self.output_mean_running.item(),
                'running_std': self.output_std_running.item(),
                'batch_means': self.output_stats['mean'][-10:],
                'batch_stds': self.output_stats['std'][-10:]
            },
            'weight_norm': self.weight_norms[-1] if self.weight_norms else 0,
            'recent_grad_norms': self.grad_norms[-10:] if self.grad_norms else [],
            'num_batches': self.num_batches_tracked.item()
        }
        return stats
    
    def reset_statistics(self):
        """Reset all tracked statistics"""
        # TODO: Reset all running statistics
        self.input_mean_running.zero_()
        self.input_std_running.fill_(1)
        self.output_mean_running.zero_()
        self.output_std_running.fill_(1)
        self.num_batches_tracked.zero_()
        
        self.input_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
        self.output_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
        self.weight_norms = []
        self.grad_norms = []

class ResidualBlock(nn.Module):
    """
    Residual block with configurable architecture:
    - Support different activation functions
    - Optional batch normalization
    - Configurable dropout
    """
    
    def __init__(self, dim, activation='relu', use_batchnorm=True, dropout=0.0):
        super(ResidualBlock, self).__init__()
        
        # TODO: Build configurable residual block
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout
        
        # First layer
        self.linear1 = nn.Linear(dim, dim)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(dim)
        
        # Second layer
        self.linear2 = nn.Linear(dim, dim)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm1d(dim)
        
        # TODO: Handle different activation functions
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            self.activation = nn.ReLU()  # Default
        
        # TODO: Add optional batch normalization
        # TODO: Include dropout if specified
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x):
        # TODO: Implement residual connection
        identity = x
        
        # First transformation
        out = self.linear1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Second transformation
        out = self.linear2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        
        # TODO: Handle dimension mismatches
        # Add residual connection
        out = out + identity
        out = self.activation(out)
        
        return out

class AttentionLayer(nn.Module):
    """
    Simple attention mechanism:
    - Compute attention weights
    - Apply to input sequence
    - Support different attention types
    """
    
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        
        # TODO: Initialize attention parameters
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # TODO: Define attention computation layers
        self.query_projection = nn.Linear(input_dim, attention_dim)
        self.key_projection = nn.Linear(input_dim, attention_dim)
        self.value_projection = nn.Linear(input_dim, attention_dim)
        
        # Output projection
        self.output_projection = nn.Linear(attention_dim, input_dim)
        
        # Scale factor for stable gradients
        self.scale = 1.0 / (attention_dim ** 0.5)
    
    def forward(self, sequence):
        # TODO: Compute attention weights
        # sequence shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = sequence.shape
        
        # Project to query, key, value
        Q = self.query_projection(sequence)  # (batch, seq_len, attention_dim)
        K = self.key_projection(sequence)    # (batch, seq_len, attention_dim)
        V = self.value_projection(sequence)  # (batch, seq_len, attention_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch, seq_len, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, seq_len, seq_len)
        
        # TODO: Apply attention to sequence
        attended_values = torch.matmul(attention_weights, V)  # (batch, seq_len, attention_dim)
        
        # Project back to input dimension
        output = self.output_projection(attended_values)  # (batch, seq_len, input_dim)
        
        # TODO: Return attended output
        return output, attention_weights

# Test custom layers
def test_custom_layers():
    """Test all custom layers"""
    batch_size, seq_len, input_dim = 4, 10, 64
    
    # Test LinearWithStats
    print("=== Testing LinearWithStats ===")
    linear_stats = LinearWithStats(input_dim, 32)
    test_input = torch.randn(batch_size, input_dim)
    
    # Forward pass in training mode
    linear_stats.train()
    output = linear_stats(test_input)
    
    # Simulate backward pass
    loss = output.sum()
    loss.backward()
    
    # Get statistics
    stats = linear_stats.get_statistics()
    print(f"Input running mean: {stats['input_stats']['running_mean']:.4f}")
    print(f"Output running mean: {stats['output_stats']['running_mean']:.4f}")
    print(f"Weight norm: {stats['weight_norm']:.4f}")
    
    # Test ResidualBlock
    print("\n=== Testing ResidualBlock ===")
    residual_block = ResidualBlock(input_dim, activation='gelu', dropout=0.1)
    residual_output = residual_block(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Residual output shape: {residual_output.shape}")
    
    # Test AttentionLayer
    print("\n=== Testing AttentionLayer ===")
    attention_layer = AttentionLayer(input_dim, attention_dim=32)
    sequence_input = torch.randn(batch_size, seq_len, input_dim)
    attended_output, attention_weights = attention_layer(sequence_input)
    print(f"Sequence input shape: {sequence_input.shape}")
    print(f"Attended output shape: {attended_output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return linear_stats, residual_block, attention_layer

# Run tests
test_custom_layers()
```

**Challenge 2: Problem-Specific Architectures**
```python
class TimeSeriesPredictor(nn.Module):
    """
    Neural network for time series prediction:
    - Handle variable sequence lengths
    - Multiple prediction horizons
    - Uncertainty estimation
    """
    
    def __init__(self, input_features, hidden_dim, output_features, sequence_length):
        super(TimeSeriesPredictor, self).__init__()
        
        # TODO: Design architecture for time series
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.output_features = output_features
        self.sequence_length = sequence_length
        
        # TODO: Consider LSTM/GRU for temporal dependencies
        self.lstm = nn.LSTM(input_features, hidden_dim, batch_first=True, num_layers=2)
        
        # TODO: Add layers for uncertainty estimation
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_features)
        )
        
        # For uncertainty estimation
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_features),
            nn.Softplus()  # Ensure positive std
        )
        
        # TODO: Handle multiple output horizons
        self.horizon_projection = nn.Linear(output_features, output_features)
    
    def forward(self, x, prediction_steps=1):
        # TODO: Process input sequence
        # x shape: (batch, sequence_length, input_features)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # TODO: Generate predictions for specified steps
        predictions = []
        uncertainties = []
        
        current_hidden = last_hidden
        for step in range(prediction_steps):
            # Predict mean and std for this step
            mean_pred = self.mean_head(current_hidden)
            std_pred = self.std_head(current_hidden) + 1e-6  # Add small epsilon
            
            predictions.append(mean_pred)
            uncertainties.append(std_pred)
            
            # Update hidden state for next prediction (simplified)
            current_hidden = F.relu(self.horizon_projection(mean_pred))
            current_hidden = torch.cat([current_hidden, last_hidden], dim=-1)
            current_hidden = current_hidden[:, :self.hidden_dim]  # Keep same dimension
        
        # TODO: Return predictions and uncertainty estimates
        mean_predictions = torch.stack(predictions, dim=1)  # (batch, prediction_steps, output_features)
        std_predictions = torch.stack(uncertainties, dim=1)   # (batch, prediction_steps, output_features)
        
        return mean_predictions, std_predictions

class ImageClassifierWithCAM(nn.Module):
    """
    Image classifier with Class Activation Maps:
    - CNN backbone for feature extraction
    - Global average pooling
    - Generate attention maps for interpretability
    """
    
    def __init__(self, num_classes, input_channels=3):
        super(ImageClassifierWithCAM, self).__init__()
        
        # TODO: Design CNN architecture
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # TODO: Add global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # TODO: Include classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # TODO: Prepare for CAM generation
        # Store feature maps for CAM
        self.feature_maps = None
        self.register_forward_hook = True
    
    def forward(self, x, return_cam=False):
        # TODO: Extract features
        feature_maps = self.features(x)  # (batch, 256, H/4, W/4)
        
        # Store feature maps for CAM generation
        if self.training or return_cam:
            self.feature_maps = feature_maps.detach()
        
        # TODO: Apply global average pooling
        pooled_features = self.global_avg_pool(feature_maps)  # (batch, 256, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # (batch, 256)
        
        # TODO: Generate classification scores
        logits = self.classifier(pooled_features)  # (batch, num_classes)
        
        # TODO: Optionally return CAM
        if return_cam:
            cam_maps = self.generate_cam_batch(logits.argmax(dim=1))
            return logits, cam_maps
        
        return logits
    
    def generate_cam(self, x, class_idx):
        """Generate Class Activation Map for specific class"""
        # TODO: Forward pass to get feature maps
        _ = self.forward(x, return_cam=False)
        
        if self.feature_maps is None:
            raise ValueError("No feature maps available. Run forward pass first.")
        
        # TODO: Weight feature maps by class weights
        # Get weights for the specified class
        class_weights = self.classifier.weight[class_idx]  # (256,)
        
        # TODO: Generate heatmap
        # feature_maps: (batch, 256, H, W)
        # class_weights: (256,)
        batch_size, num_channels, height, width = self.feature_maps.shape
        
        # Weighted combination of feature maps
        cam = torch.zeros(batch_size, height, width)
        for i in range(num_channels):
            cam += class_weights[i] * self.feature_maps[:, i, :, :]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        for b in range(batch_size):
            cam_min = cam[b].min()
            cam_max = cam[b].max()
            if cam_max > cam_min:
                cam[b] = (cam[b] - cam_min) / (cam_max - cam_min)
        
        return cam
    
    def generate_cam_batch(self, class_indices):
        """Generate CAM for multiple samples with different classes"""
        if self.feature_maps is None:
            raise ValueError("No feature maps available.")
        
        batch_size, num_channels, height, width = self.feature_maps.shape
        cam_maps = torch.zeros(batch_size, height, width)
        
        for b in range(batch_size):
            class_idx = class_indices[b]
            class_weights = self.classifier.weight[class_idx]
            
            # Generate CAM for this sample
            cam = torch.zeros(height, width)
            for i in range(num_channels):
                cam += class_weights[i] * self.feature_maps[b, i, :, :]
            
            cam = F.relu(cam)
            
            # Normalize
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            
            cam_maps[b] = cam
        
        return cam_maps

class MultiTaskNetwork(nn.Module):
    """
    Multi-task learning network:
    - Shared backbone
    - Task-specific heads
    - Loss balancing mechanisms
    """
    
    def __init__(self, input_dim, shared_dim, task_configs):
        super(MultiTaskNetwork, self).__init__()
        
        # TODO: Create shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, shared_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(shared_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.BatchNorm1d(shared_dim)
        )
        
        # TODO: Create task-specific heads
        self.task_heads = nn.ModuleDict()
        self.task_names = list(task_configs.keys())
        
        for task_name, config in task_configs.items():
            task_head = nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(shared_dim // 2, config['output_dim'])
            )
            self.task_heads[task_name] = task_head
        
        # TODO: Initialize loss balancing parameters
        self.task_weights = nn.Parameter(torch.ones(len(task_configs)))
        self.register_buffer('task_losses', torch.zeros(len(task_configs)))
        
    def forward(self, x, active_tasks=None):
        # TODO: Process through shared backbone
        shared_features = self.shared_backbone(x)  # (batch, shared_dim)
        
        # TODO: Apply task-specific heads
        if active_tasks is None:
            active_tasks = self.task_names
        
        task_outputs = {}
        for task_name in active_tasks:
            if task_name in self.task_heads:
                task_output = self.task_heads[task_name](shared_features)
                task_outputs[task_name] = task_output
        
        # TODO: Return task-specific outputs
        return task_outputs
    
    def compute_balanced_loss(self, outputs, targets, loss_functions):
        """Compute weighted multi-task loss"""
        total_loss = 0
        individual_losses = {}
        
        for i, task_name in enumerate(self.task_names):
            if task_name in outputs and task_name in targets:
                task_loss = loss_functions[task_name](outputs[task_name], targets[task_name])
                
                # Apply learned weight
                weighted_loss = self.task_weights[i] * task_loss
                total_loss += weighted_loss
                
                individual_losses[task_name] = task_loss.item()
                
                # Update running average of task losses
                self.task_losses[i] = 0.9 * self.task_losses[i] + 0.1 * task_loss.item()
        
        return total_loss, individual_losses
    
    def update_task_weights(self):
        """Update task weights based on relative loss magnitudes"""
        with torch.no_grad():
            # Inverse relationship: higher loss -> lower weight
            avg_loss = self.task_losses.mean()
            relative_losses = self.task_losses / (avg_loss + 1e-8)
            
            # Update weights (inverse relationship)
            self.task_weights.data = 1.0 / (relative_losses + 1e-8)
            
            # Normalize weights
            self.task_weights.data = self.task_weights.data / self.task_weights.data.sum() * len(self.task_names)

# Test problem-specific architectures
def test_architectures():
    """Test all problem-specific architectures"""
    
    # Test TimeSeriesPredictor
    print("=== Testing TimeSeriesPredictor ===")
    ts_model = TimeSeriesPredictor(input_features=5, hidden_dim=64, output_features=1, sequence_length=20)
    ts_input = torch.randn(8, 20, 5)  # batch=8, seq_len=20, features=5
    
    mean_preds, std_preds = ts_model(ts_input, prediction_steps=5)
    print(f"Time series input shape: {ts_input.shape}")
    print(f"Mean predictions shape: {mean_preds.shape}")
    print(f"Std predictions shape: {std_preds.shape}")
    
    # Test ImageClassifierWithCAM
    print("\n=== Testing ImageClassifierWithCAM ===")
    img_model = ImageClassifierWithCAM(num_classes=10, input_channels=3)
    img_input = torch.randn(4, 3, 64, 64)  # batch=4, channels=3, 64x64 images
    
    logits, cam_maps = img_model(img_input, return_cam=True)
    print(f"Image input shape: {img_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"CAM maps shape: {cam_maps.shape}")
    
    # Test MultiTaskNetwork
    print("\n=== Testing MultiTaskNetwork ===")
    task_configs = {
        'regression': {'output_dim': 1},
        'classification': {'output_dim': 5},
        'ranking': {'output_dim': 3}
    }
    
    mt_model = MultiTaskNetwork(input_dim=50, shared_dim=128, task_configs=task_configs)
    mt_input = torch.randn(16, 50)  # batch=16, features=50
    
    outputs = mt_model(mt_input)
    print(f"Multi-task input shape: {mt_input.shape}")
    for task_name, output in outputs.items():
        print(f"{task_name} output shape: {output.shape}")
    
    return ts_model, img_model, mt_model

# Run architecture tests
test_architectures()
```

#### Evaluation Criteria
- [ ] Implements complex architectures correctly
- [ ] Demonstrates understanding of design principles
- [ ] Code is modular and reusable
- [ ] Includes proper error handling

---

## Week 5: Training Loop Patterns

### Learning Objectives
- Memorize standard training loop syntax
- Understand optimizer and loss function usage
- Implement advanced training techniques

---

### Exercise 5.1: Basic Training Loop Syntax
**Type**: Muscle Memory  
**Time**: 4-5 hours  
**Difficulty**: Intermediate

#### Objective
Memorize complete training loop pattern through repetition.

#### Requirements
Write the complete training loop **EXACTLY 20 times** with variations following these specific requirements:

**TEMPLATE PATTERNS TO FOLLOW:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Template for Basic Training Loop:
def basic_training_template():
    # 1. Setup components
    model = ModelClass()
    criterion = LossFunction()
    optimizer = OptimizerClass(model.parameters(), lr=learning_rate)
    
    # 2. Data preparation
    dataset = DatasetClass()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Training loop structure
    num_epochs = num_epochs_value
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            # CRITICAL 5-STEP PATTERN:
            optimizer.zero_grad()          # Step 1: Zero gradients
            output = model(data)           # Step 2: Forward pass
            loss = criterion(output, target) # Step 3: Compute loss
            loss.backward()                # Step 4: Backward pass
            optimizer.step()               # Step 5: Update parameters
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Template for Training with Validation:
def training_with_validation_template():
    # Training phase
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
```

**5-STEP TRAINING PATTERN - MEMORIZE EXACTLY:**
1. `optimizer.zero_grad()`
2. `output = model(data)`
3. `loss = criterion(output, target)`
4. `loss.backward()`
5. `optimizer.step()`

**Practice Set 1: Basic Regression Loops - 5 REQUIRED variants**
- Variant 1: Linear regression - MSELoss, Adam optimizer, lr=0.001, 10 epochs
- Variant 2: Polynomial regression - MSELoss, SGD optimizer, lr=0.01, 15 epochs
- Variant 3: Multi-output regression - MSELoss, RMSprop optimizer, lr=0.0005, 20 epochs
- Variant 4: Robust regression - L1Loss, Adam optimizer, lr=0.002, 12 epochs
- Variant 5: Regularized regression - MSELoss + L2 penalty, AdamW optimizer, lr=0.001, 25 epochs

**Practice Set 2: Classification Loops - 5 REQUIRED variants**
- Variant 6: Binary classification - BCEWithLogitsLoss, Adam, lr=0.001, sigmoid output
- Variant 7: Multi-class classification - CrossEntropyLoss, SGD, lr=0.01, 10 classes
- Variant 8: MNIST classification - NLLLoss + LogSoftmax, Adam, lr=0.001, 10 classes
- Variant 9: Imbalanced classification - CrossEntropyLoss with weights, Adam, lr=0.0005
- Variant 10: Multi-label classification - BCEWithLogitsLoss, AdamW, lr=0.001, multiple outputs

**Practice Set 3: Training with Validation - 5 REQUIRED variants**
- Variant 11: Standard train/val split - 80/20, monitor both losses
- Variant 12: K-fold cross validation setup - different folds, average metrics
- Variant 13: Early stopping - monitor validation loss, stop if no improvement
- Variant 14: Best model saving - save model when validation loss improves
- Variant 15: Learning rate scheduling - reduce LR when validation plateaus

**Practice Set 4: Advanced Training Patterns - 5 REQUIRED variants**
- Variant 16: Gradient clipping - clip gradients by norm, max_norm=1.0
- Variant 17: Mixed precision training - use autocast và GradScaler
- Variant 18: Accumulate gradients - effective larger batch size
- Variant 19: Curriculum learning - start with easy samples, increase difficulty
- Variant 20: Multi-task training - multiple losses, different weights

**Specific Requirements for Each Variant:**

**Variant 1: Linear Regression**
```python
# Model: nn.Linear(1, 1)
# Loss: nn.MSELoss()
# Optimizer: optim.Adam(lr=0.001)
# Data: y = 2*x + 1 + noise
# Epochs: 10
# Batch size: 32
```

**Variant 6: Binary Classification**
```python
# Model: nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))
# Loss: nn.BCEWithLogitsLoss()
# Optimizer: optim.Adam(lr=0.001)
# Data: 2D points, binary labels
# Epochs: 15
# Batch size: 64
```

**Variant 11: Train/Val Split**
```python
# Same as variant 6 but add:
# - Split data 80/20
# - model.train() for training
# - model.eval() for validation
# - torch.no_grad() for validation
# - Track both train_loss và val_loss
```

**Variant 16: Gradient Clipping**
```python
# Add after loss.backward():
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Then optimizer.step()
```

**Daily Practice Schedule:**
- **Day 1**: Variants 1-5 (Regression loops) - 5 times each
- **Day 2**: Variants 6-10 (Classification loops) - 5 times each
- **Day 3**: Variants 11-15 (Validation loops) - 5 times each
- **Day 4**: Variants 16-20 (Advanced patterns) - 5 times each
- **Day 5**: Mixed practice - random variants
- **Day 6**: Speed practice - write from memory
- **Day 7**: Error handling và debugging practice

**CRITICAL**: You must write ALL 20 training loops. 5-step pattern must become muscle memory!

// ... existing code ...
```

#### Deliverables
- **training_loops.py**: 20 different training loop implementations
- **loop_checklist.md**: Step-by-step training loop checklist

---

### Exercise 5.2: Advanced Training Applications
**Type**: Thinking + Application  
**Time**: 6-7 hours  
**Difficulty**: Intermediate-Advanced

#### Objective
Implement training loops for complex scenarios and challenges.

#### Requirements

**Scenario 1: Multi-Task Training**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MultiTaskTrainer:
    """
    Trainer for multi-task learning scenarios:
    - Handle multiple losses
    - Balance task importance
    - Monitor task-specific metrics
    """
    
    def __init__(self, model, task_configs, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # TODO: Initialize task-specific optimizers/schedulers
        self.optimizers = {}
        self.schedulers = {}
        self.loss_functions = {}
        
        for task_name, config in task_configs.items():
            # Separate optimizer for each task (optional)
            self.optimizers[task_name] = optim.Adam(
                model.parameters(), 
                lr=config.get('lr', 0.001),
                weight_decay=config.get('weight_decay', 0.0)
            )
            
            # Learning rate scheduler
            self.schedulers[task_name] = optim.lr_scheduler.StepLR(
                self.optimizers[task_name], 
                step_size=config.get('step_size', 50), 
                gamma=config.get('gamma', 0.5)
            )
            
            # Loss function
            if config['task_type'] == 'classification':
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
            elif config['task_type'] == 'regression':
                self.loss_functions[task_name] = nn.MSELoss()
            else:
                self.loss_functions[task_name] = nn.MSELoss()  # Default
        
        # TODO: Set up loss balancing mechanisms
        self.task_weights = {name: 1.0 for name in self.task_names}
        self.loss_history = {name: [] for name in self.task_names}
        self.uncertainty_weights = nn.Parameter(torch.ones(len(self.task_names)))
        
        # TODO: Initialize metric tracking
        self.metrics_history = {name: {'loss': [], 'accuracy': []} for name in self.task_names}
        
    def train_epoch(self, data_loaders):
        """Train one epoch with multiple tasks"""
        self.model.train()
        
        # TODO: Handle multiple data loaders
        # TODO: Balance sampling between tasks
        # TODO: Compute task-specific losses
        # TODO: Implement loss balancing strategy
        # TODO: Update model parameters
        
        # Create iterators for each task
        task_iterators = {}
        task_data_remaining = {}
        
        for task_name, loader in data_loaders.items():
            task_iterators[task_name] = iter(loader)
            task_data_remaining[task_name] = len(loader)
        
        epoch_losses = {name: 0.0 for name in self.task_names}
        epoch_samples = {name: 0 for name in self.task_names}
        
        # Training loop with round-robin sampling
        total_batches = max(task_data_remaining.values())
        
        for batch_idx in range(total_batches):
            batch_losses = {}
            
            # Sample from each task
            for task_name in self.task_names:
                if task_data_remaining[task_name] > 0:
                    try:
                        # Get next batch
                        data, target = next(task_iterators[task_name])
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(data, active_tasks=[task_name])
                        task_output = outputs[task_name]
                        
                        # Compute loss
                        loss = self.loss_functions[task_name](task_output, target)
                        batch_losses[task_name] = loss
                        
                        epoch_losses[task_name] += loss.item()
                        epoch_samples[task_name] += data.size(0)
                        task_data_remaining[task_name] -= 1
                        
                    except StopIteration:
                        # No more data for this task
                        task_data_remaining[task_name] = 0
                        continue
            
            # Compute combined loss
            if batch_losses:
                combined_loss = self.adaptive_loss_balancing(batch_losses)
                
                # Backward pass
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                
                combined_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters (use one optimizer for simplicity)
                list(self.optimizers.values())[0].step()
        
        # Calculate average losses
        avg_losses = {}
        for task_name in self.task_names:
            if epoch_samples[task_name] > 0:
                avg_losses[task_name] = epoch_losses[task_name] / len(data_loaders[task_name])
                self.loss_history[task_name].append(avg_losses[task_name])
        
        return avg_losses
    
    def validate(self, val_loaders):
        """Validate on all tasks"""
        self.model.eval()
        
        # TODO: Evaluate each task separately
        # TODO: Compute task-specific metrics
        # TODO: Return comprehensive results
        
        val_results = {}
        
        with torch.no_grad():
            for task_name, val_loader in val_loaders.items():
                task_loss = 0.0
                task_correct = 0
                task_total = 0
                
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(data, active_tasks=[task_name])
                    task_output = outputs[task_name]
                    
                    # Compute loss
                    loss = self.loss_functions[task_name](task_output, target)
                    task_loss += loss.item()
                    
                    # Compute accuracy (for classification tasks)
                    if self.task_configs[task_name]['task_type'] == 'classification':
                        _, predicted = torch.max(task_output, 1)
                        task_total += target.size(0)
                        task_correct += (predicted == target).sum().item()
                
                # Calculate metrics
                avg_loss = task_loss / len(val_loader)
                accuracy = task_correct / task_total if task_total > 0 else 0.0
                
                val_results[task_name] = {
                    'loss': avg_loss,
                    'accuracy': accuracy
                }
                
                # Update metrics history
                self.metrics_history[task_name]['loss'].append(avg_loss)
                self.metrics_history[task_name]['accuracy'].append(accuracy)
        
        return val_results
    
    def adaptive_loss_balancing(self, task_losses):
        """Implement adaptive loss balancing"""
        # TODO: Implement uncertainty-based balancing
        # TODO: Or gradient-based balancing
        # TODO: Update task weights dynamically
        
        # Method 1: Uncertainty-based weighting (simplified)
        weighted_losses = []
        for i, (task_name, loss) in enumerate(task_losses.items()):
            # Use learnable uncertainty weights
            uncertainty = self.uncertainty_weights[i]
            weighted_loss = loss / (2 * uncertainty**2) + torch.log(uncertainty)
            weighted_losses.append(weighted_loss)
        
        # Method 2: Dynamic task weighting based on loss magnitude
        if len(self.loss_history[list(task_losses.keys())[0]]) > 0:
            # Calculate relative loss changes
            for task_name, loss in task_losses.items():
                recent_losses = self.loss_history[task_name][-10:]  # Last 10 epochs
                if len(recent_losses) > 1:
                    avg_recent_loss = sum(recent_losses) / len(recent_losses)
                    
                    # Adjust weight based on loss trend
                    if loss.item() > avg_recent_loss:
                        self.task_weights[task_name] *= 1.1  # Increase weight for struggling tasks
                    else:
                        self.task_weights[task_name] *= 0.99  # Slightly decrease weight
                    
                    # Clip weights
                    self.task_weights[task_name] = max(0.1, min(2.0, self.task_weights[task_name]))
        
        # Combine losses with current weights
        total_loss = sum(self.task_weights[task_name] * loss 
                        for task_name, loss in task_losses.items())
        
        return total_loss
    
    def update_learning_rates(self):
        """Update learning rates for all tasks"""
        for scheduler in self.schedulers.values():
            scheduler.step()

def curriculum_learning():
    """
    Implement curriculum learning:
    - Start with easier examples
    - Gradually increase difficulty
    - Monitor learning progress
    """
    
    def difficulty_scorer(batch_data, batch_labels, model):
        """Score batch difficulty (problem-specific)"""
        # TODO: Implement difficulty metric
        # TODO: Consider label noise, complexity, etc.
        
        with torch.no_grad():
            model.eval()
            outputs = model(batch_data)
            
            # Method 1: Use prediction confidence as difficulty measure
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            
            # Lower confidence = higher difficulty
            difficulty_scores = 1.0 - max_probs
            
            # Method 2: Use loss as difficulty measure
            criterion = nn.CrossEntropyLoss(reduction='none')
            individual_losses = criterion(outputs, batch_labels)
            
            # Combine both measures
            combined_difficulty = 0.7 * difficulty_scores + 0.3 * individual_losses
            
            return combined_difficulty
    
    def create_curriculum_schedule(dataset, model, num_epochs):
        """Create curriculum schedule"""
        # TODO: Sort data by difficulty
        # TODO: Create schedule for introducing harder examples
        # TODO: Return epoch-wise data subsets
        
        # Score all samples
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        all_difficulties = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                difficulties = difficulty_scorer(data, target, model)
                all_difficulties.extend(difficulties.tolist())
                
                # Track original indices
                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + data.size(0)
                all_indices.extend(list(range(start_idx, end_idx)))
        
        # Sort by difficulty
        sorted_pairs = sorted(zip(all_difficulties, all_indices))
        sorted_difficulties, sorted_indices = zip(*sorted_pairs)
        
        # Create curriculum schedule
        total_samples = len(sorted_indices)
        curriculum_schedule = {}
        
        for epoch in range(num_epochs):
            # Gradually introduce more difficult samples
            if epoch < num_epochs // 3:
                # Easy phase: use easiest 40% of data
                num_samples = int(0.4 * total_samples)
            elif epoch < 2 * num_epochs // 3:
                # Medium phase: use easiest 70% of data
                num_samples = int(0.7 * total_samples)
            else:
                # Hard phase: use all data
                num_samples = total_samples
            
            # Select samples for this epoch
            epoch_indices = sorted_indices[:num_samples]
            curriculum_schedule[epoch] = epoch_indices
        
        return curriculum_schedule
    
    def curriculum_training_loop(model, full_dataset, val_loader, num_epochs=100):
        """Training loop with curriculum learning"""
        # TODO: Implement curriculum-based training
        # TODO: Gradually increase data difficulty
        # TODO: Monitor performance improvements
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create curriculum schedule
        curriculum_schedule = create_curriculum_schedule(full_dataset, model, num_epochs)
        
        training_history = {'epoch': [], 'num_samples': [], 'train_loss': [], 'val_accuracy': []}
        
        for epoch in range(num_epochs):
            # Get samples for this epoch
            epoch_indices = curriculum_schedule[epoch]
            
            # Create subset dataset
            subset_data = torch.utils.data.Subset(full_dataset, epoch_indices)
            train_loader = DataLoader(subset_data, batch_size=32, shuffle=True)
            
            # Training phase
            model.train()
            epoch_loss = 0.0
            
            for data, target in train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_accuracy = 100 * correct / total
            
            # Record progress
            training_history['epoch'].append(epoch)
            training_history['num_samples'].append(len(epoch_indices))
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Samples={len(epoch_indices)}, "
                      f"Train Loss={avg_train_loss:.4f}, Val Acc={val_accuracy:.2f}%")
        
        return training_history
    
    return curriculum_training_loop
```

**Scenario 2: Robust Training Techniques**
```python
def robust_training_implementations():
    """
    Implement robust training techniques:
    - Handle gradient explosion/vanishing
    - Implement early stopping variants
    - Add noise injection for robustness
    """
    
    class RobustTrainer:
        def __init__(self, model, optimizer, criterion):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            
            # TODO: Initialize gradient tracking
            self.gradient_norms = []
            self.loss_history = []
            self.patience_counter = 0
            self.best_val_loss = float('inf')
            self.best_model_state = None
            
            # TODO: Set up early stopping parameters
            self.early_stopping_patience = 10
            self.min_delta = 1e-4
            
            # TODO: Configure noise injection
            self.noise_std = 0.01
            self.noise_schedule = lambda epoch: max(0.001, self.noise_std * (0.95 ** epoch))
            
        def train_step_with_gradient_monitoring(self, data, target):
            """Training step with gradient monitoring"""
            
            # TODO: Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # TODO: Compute loss
            # TODO: Check loss for anomalies
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected: {loss.item()}")
                return None
            
            # TODO: Backward pass
            loss.backward()
            
            # TODO: Monitor gradient norms
            total_grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.norm(2)
                    total_grad_norm += param_grad_norm.item() ** 2
            
            total_grad_norm = total_grad_norm ** 0.5
            self.gradient_norms.append(total_grad_norm)
            
            # TODO: Apply gradient clipping if needed
            # TODO: Check for gradient explosion/vanishing
            if total_grad_norm > 10.0:  # Gradient explosion
                print(f"Warning: Large gradient norm detected: {total_grad_norm:.4f}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            elif total_grad_norm < 1e-6:  # Vanishing gradients
                print(f"Warning: Very small gradient norm: {total_grad_norm:.8f}")
            
            # TODO: Update parameters
            self.optimizer.step()
            
            # TODO: Log gradient statistics
            self.loss_history.append(loss.item())
            
            return {
                'loss': loss.item(),
                'grad_norm': total_grad_norm,
                'grad_clipped': total_grad_norm > 10.0
            }
        
        def adaptive_early_stopping(self, val_loss, patience=None):
            """Adaptive early stopping based on multiple criteria"""
            # TODO: Track validation loss trends
            if patience is None:
                patience = self.early_stopping_patience
            
            # TODO: Consider loss oscillations
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                return False  # Continue training
            else:
                self.patience_counter += 1
            
            # TODO: Implement adaptive patience
            # Increase patience if loss is oscillating
            if len(self.loss_history) > 10:
                recent_losses = self.loss_history[-10:]
                loss_variance = torch.var(torch.tensor(recent_losses))
                
                if loss_variance > 0.1:  # High variance indicates oscillation
                    effective_patience = patience * 1.5
                else:
                    effective_patience = patience
            else:
                effective_patience = patience
            
            # TODO: Check for overfitting patterns
            should_stop = self.patience_counter >= effective_patience
            
            if should_stop:
                print(f"Early stopping triggered. Best val loss: {self.best_val_loss:.6f}")
                if self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
            
            return should_stop
        
        def noise_injection_training(self, data, epoch, noise_std=None):
            """Add noise for robustness"""
            # TODO: Add input noise
            if noise_std is None:
                noise_std = self.noise_schedule(epoch)
            
            # Add Gaussian noise to input
            input_noise = torch.randn_like(data) * noise_std
            noisy_data = data + input_noise
            
            # TODO: Add weight noise (optional)
            if torch.rand(1).item() < 0.1:  # 10% chance to add weight noise
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.requires_grad:
                            weight_noise = torch.randn_like(param) * (noise_std * 0.1)
                            param.add_(weight_noise)
            
            # TODO: Schedule noise reduction over time
            return noisy_data
        
        def get_training_diagnostics(self):
            """Get comprehensive training diagnostics"""
            diagnostics = {
                'gradient_norms': {
                    'mean': sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0,
                    'max': max(self.gradient_norms) if self.gradient_norms else 0,
                    'min': min(self.gradient_norms) if self.gradient_norms else 0,
                    'recent_10': self.gradient_norms[-10:] if len(self.gradient_norms) >= 10 else self.gradient_norms
                },
                'loss_history': {
                    'recent_10': self.loss_history[-10:] if len(self.loss_history) >= 10 else self.loss_history,
                    'trend': 'decreasing' if len(self.loss_history) > 5 and 
                            self.loss_history[-1] < self.loss_history[-5] else 'stable/increasing'
                },
                'early_stopping': {
                    'patience_counter': self.patience_counter,
                    'best_val_loss': self.best_val_loss,
                    'has_best_model': self.best_model_state is not None
                }
            }
            return diagnostics

def adversarial_training():
    """
    Implement adversarial training:
    - Generate adversarial examples
    - Train on mixed clean/adversarial data
    - Monitor robustness metrics
    """
    
    def generate_adversarial_examples(model, data, target, epsilon=0.1):
        """Generate adversarial examples using FGSM"""
        # TODO: Compute gradients w.r.t. input
        data.requires_grad_(True)
        
        outputs = model(data)
        loss = F.cross_entropy(outputs, target)
        
        # TODO: Generate adversarial perturbations
        model.zero_grad()
        loss.backward()
        
        # Get gradient sign
        data_grad = data.grad.sign()
        
        # TODO: Apply perturbations to create adversarial examples
        adversarial_data = data + epsilon * data_grad
        
        # Clamp to valid input range (assuming [0, 1])
        adversarial_data = torch.clamp(adversarial_data, 0, 1)
        
        return adversarial_data.detach()
    
    def adversarial_training_loop(model, train_loader, val_loader, num_epochs=50, epsilon=0.1):
        """Training loop with adversarial examples"""
        # TODO: Generate adversarial examples each batch
        # TODO: Mix clean and adversarial data
        # TODO: Train on mixed dataset
        # TODO: Evaluate robustness periodically
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        training_history = {
            'epoch': [], 'clean_acc': [], 'adv_acc': [], 'train_loss': []
        }
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Generate adversarial examples
                adversarial_data = generate_adversarial_examples(model, data, target, epsilon)
                
                # Mix clean and adversarial data
                if torch.rand(1).item() < 0.5:
                    # Use adversarial examples
                    mixed_data = adversarial_data
                else:
                    # Use clean examples
                    mixed_data = data
                
                # Forward pass
                outputs = model(mixed_data)
                loss = criterion(outputs, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Evaluate on clean and adversarial examples
            if epoch % 5 == 0:
                clean_acc = evaluate_accuracy(model, val_loader)
                adv_acc = evaluate_adversarial_accuracy(model, val_loader, epsilon)
                
                training_history['epoch'].append(epoch)
                training_history['clean_acc'].append(clean_acc)
                training_history['adv_acc'].append(adv_acc)
                training_history['train_loss'].append(epoch_loss / len(train_loader))
                
                print(f"Epoch {epoch}: Train Loss={epoch_loss/len(train_loader):.4f}, "
                      f"Clean Acc={clean_acc:.2f}%, Adv Acc={adv_acc:.2f}%")
        
        return training_history
    
    def evaluate_accuracy(model, data_loader):
        """Evaluate clean accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100 * correct / total
    
    def evaluate_adversarial_accuracy(model, data_loader, epsilon):
        """Evaluate robustness against adversarial examples"""
        model.eval()
        correct = 0
        total = 0
        
        for data, target in data_loader:
            # Generate adversarial examples
            adversarial_data = generate_adversarial_examples(model, data, target, epsilon)
            
            with torch.no_grad():
                outputs = model(adversarial_data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100 * correct / total
    
    return adversarial_training_loop, evaluate_adversarial_accuracy
```

**Scenario 3: Distributed Training Simulation**
```python
def distributed_training_concepts():
    """
    Implement concepts from distributed training:
    - Gradient accumulation for large batch simulation
    - Synchronization mechanisms
    - Communication overhead simulation
    """
    
    class DistributedSimulator:
        def __init__(self, model, num_workers=4, accumulation_steps=4):
            self.model = model
            self.num_workers = num_workers
            self.accumulation_steps = accumulation_steps
            
        def gradient_accumulation_training(self, dataloader, optimizer, num_epochs=10):
            """Simulate large batch training with gradient accumulation"""
            
            # TODO: Accumulate gradients over multiple mini-batches
            # TODO: Scale gradients appropriately
            # TODO: Update parameters after accumulation
            # TODO: Simulate memory constraints
            
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                accumulated_samples = 0
                
                for batch_idx, (data, target) in enumerate(dataloader):
                    # Forward pass
                    outputs = self.model(data)
                    loss = criterion(outputs, target)
                    
                    # Scale loss by accumulation steps
                    scaled_loss = loss / self.accumulation_steps
                    scaled_loss.backward()
                    
                    epoch_loss += loss.item()
                    accumulated_samples += data.size(0)
                    
                    # Update parameters after accumulation
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        # Simulate gradient synchronization
                        self.simulated_all_reduce()
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        print(f"Updated parameters after {accumulated_samples} samples")
                        accumulated_samples = 0
                
                print(f"Epoch {epoch}: Average Loss = {epoch_loss / len(dataloader):.4f}")
        
        def simulated_all_reduce(self, communication_delay=0.001):
            """Simulate all-reduce operation"""
            # TODO: Simulate communication delays
            import time
            time.sleep(communication_delay)
            
            # TODO: Average gradients across workers
            # In real distributed training, this would average gradients from all workers
            # Here we simulate by adding small noise to represent communication imperfection
            
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        # TODO: Add communication noise (optional)
                        noise = torch.randn_like(param.grad) * 1e-6
                        param.grad.add_(noise)
                        
                        # Simulate averaging (in real scenario, this would be actual averaging)
                        param.grad.mul_(1.0)  # No change for single worker simulation
        
        def asynchronous_updates(self, dataloader, optimizers, staleness_probability=0.1):
            """Simulate asynchronous parameter updates"""
            # TODO: Implement delayed gradient application
            # TODO: Handle staleness in gradients
            # TODO: Monitor convergence differences
            
            gradient_buffer = []
            criterion = nn.CrossEntropyLoss()
            
            for batch_idx, (data, target) in enumerate(dataloader):
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, target)
                
                # Backward pass
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                loss.backward()
                
                # Store gradients with timestamp
                current_gradients = {}
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        current_gradients[name] = param.grad.clone()
                
                gradient_buffer.append({
                    'gradients': current_gradients,
                    'timestamp': batch_idx,
                    'applied': False
                })
                
                # Apply gradients with possible staleness
                for grad_info in gradient_buffer:
                    if not grad_info['applied']:
                        # Simulate staleness
                        if torch.rand(1).item() > staleness_probability:
                            # Apply this gradient update
                            for name, param in self.model.named_parameters():
                                if name in grad_info['gradients']:
                                    param.grad = grad_info['gradients'][name]
                            
                            optimizers[0].step()  # Use first optimizer
                            grad_info['applied'] = True
                            
                            staleness = batch_idx - grad_info['timestamp']
                            if staleness > 0:
                                print(f"Applied stale gradient (staleness: {staleness})")
                
                # Clean up applied gradients
                gradient_buffer = [g for g in gradient_buffer if not g['applied']]
    
    # Create sample data and model for testing
    def test_distributed_concepts():
        # Sample model and data
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )
        
        # Sample dataset
        X = torch.randn(1000, 10)
        y = torch.randint(0, 5, (1000,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Test distributed simulator
        simulator = DistributedSimulator(model, num_workers=4, accumulation_steps=4)
        
        print("=== Testing Gradient Accumulation ===")
        simulator.gradient_accumulation_training(dataloader, optimizer, num_epochs=3)
        
        print("\n=== Testing Asynchronous Updates ===")
        optimizers = [optim.SGD(model.parameters(), lr=0.01) for _ in range(4)]
        simulator.asynchronous_updates(dataloader, optimizers, staleness_probability=0.2)
        
        return simulator
    
    return test_distributed_concepts

# Test all advanced training scenarios
def test_advanced_training():
    """Test all advanced training implementations"""
    
    # Create sample multi-task data
    def create_multitask_data():
        # Shared input features
        X = torch.randn(1000, 50)
        
        # Task 1: Regression
        y_reg = torch.sum(X[:, :10], dim=1, keepdim=True) + 0.1 * torch.randn(1000, 1)
        
        # Task 2: Classification
        y_cls = (torch.sum(X[:, 10:20], dim=1) > 0).long()
        
        # Create datasets
        reg_dataset = TensorDataset(X, y_reg)
        cls_dataset = TensorDataset(X, y_cls)
        
        return {
            'regression': DataLoader(reg_dataset, batch_size=32, shuffle=True),
            'classification': DataLoader(cls_dataset, batch_size=32, shuffle=True)
        }
    
    # Sample multi-task model (using previously defined MultiTaskNetwork)
    task_configs = {
        'regression': {'task_type': 'regression', 'lr': 0.001},
        'classification': {'task_type': 'classification', 'lr': 0.001}
    }
    
    model = MultiTaskNetwork(
        input_dim=50, 
        shared_dim=64, 
        task_configs={
            'regression': {'output_dim': 1},
            'classification': {'output_dim': 2}
        }
    )
    
    # Test multi-task training
    print("=== Testing Multi-Task Training ===")
    trainer = MultiTaskTrainer(model, task_configs)
    train_loaders = create_multitask_data()
    val_loaders = create_multitask_data()  # Same for simplicity
    
    # Train for a few epochs
    for epoch in range(3):
        train_losses = trainer.train_epoch(train_loaders)
        val_results = trainer.validate(val_loaders)
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train losses: {train_losses}")
        print(f"  Val results: {val_results}")
        
        trainer.update_learning_rates()
    
    # Test other training methods
    print("\n=== Testing Distributed Training Concepts ===")
    distributed_test = distributed_training_concepts()
    distributed_test()
    
    return trainer

# Run all tests
test_advanced_training()
```

#### Evaluation Criteria
- [ ] Handles complex training scenarios correctly
- [ ] Implements robust training techniques
- [ ] Monitors and reports appropriate metrics
- [ ] Code is well-structured and documented

---

## Week 6: Integration & Mastery Assessment

### Learning Objectives
- Integrate all learned concepts into complete projects
- Demonstrate mastery through timed challenges
- Build portfolio-worthy implementations

---

### Exercise 6.1: Speed Coding Challenge
**Type**: Timed Muscle Memory Test  
**Time**: 2 hours  
**Difficulty**: Intermediate

#### Objective
Write common PyTorch patterns from memory within time limits.

#### Requirements
Complete timed challenges to test muscle memory mastery:

**TEMPLATE PATTERNS FOR NLP SPEED CODING:**

```python
# Template for NLP Models (timed challenge):
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1. NLP Model Definition Template:
class NLPModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(NLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state for classification
        output = self.fc(hidden[-1])  # (batch_size, num_classes)
        return output

# 2. Text Preprocessing Template:
def tokenize_text(texts, vocab):
    tokenized = []
    for text in texts:
        tokens = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
        tokenized.append(torch.tensor(tokens))
    return tokenized

def collate_fn(batch):
    # Handle variable length sequences
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return texts, labels

# 3. Training Setup Template:
model = NLPModel(vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. NLP Training Loop Template:
for epoch in range(num_epochs):
    for batch_texts, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_texts)  # (batch_size, num_classes)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        # Gradient clipping for RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

**Challenge 1: Basic Neural Network (10 minutes)**
**Requirements:**
- Model architecture: 784 → 128 → 64 → 10 (3 layers)
- Use ReLU activations between layers
- Add Dropout(0.2) after each hidden layer
- Setup Adam optimizer with lr=0.001
- Write complete training loop structure (5-step pattern)
- Include validation phase with model.eval() and torch.no_grad()
- Print epoch progress
- **Time limit: 10 minutes**
- **Must write from memory without documentation**

**Challenge 2: Text Classifier with Embeddings (15 minutes)**  
**Requirements:**
- Embedding layer: nn.Embedding(vocab_size=10000, embed_dim=128)
- LSTM layer: nn.LSTM(128, 64, batch_first=True)
- FC layers: 64 → 32 → num_classes
- Handle variable length sequences correctly
- Use nn.utils.rnn.pad_sequence for batching
- Setup training loop for text classification
- Include both training và validation phases
- Add proper sequence length handling
- **Time limit: 15 minutes**
- **Must handle LSTM output correctly (last hidden state)**

**Challenge 3: Sentiment Analysis Pipeline (20 minutes)**
**Requirements:**
- Complete sentiment analysis system (positive/negative reviews)
- Text preprocessing: tokenization, vocabulary building, numericalization
- Model: Embedding → BiLSTM → Attention → FC → Sigmoid
- Include train/validation data split
- Implement validation loop with accuracy and F1-score calculation
- Add model saving: `torch.save(model.state_dict(), 'sentiment_model.pth')`
- Add model loading: `model.load_state_dict(torch.load('sentiment_model.pth'))`
- Include text preprocessing utilities
- Add early stopping mechanism based on validation F1-score
- Proper sequence padding và attention masking
- **Time limit: 20 minutes**
- **Must handle variable-length sequences correctly**

**Challenge 4: Named Entity Recognition (NER) System (25 minutes)**
**Requirements:**
- Token-level classification: word → entity tag (B-PER, I-PER, B-LOC, etc.)
- Model: Embedding → BiLSTM → CRF layer (or linear layer for simplicity)
- Handle BIO tagging scheme correctly
- Learning rate scheduler: `optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)`
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Custom NER loss function (handle padding tokens)
- Tensorboard logging for precision/recall per entity type
- Model checkpointing every 5 epochs
- Resume training from checkpoint
- **Time limit: 25 minutes**
- **Token-level sequence labeling expertise**

**Challenge 5: Transformer Language Model (30 minutes)**
**Requirements:**
- Implement mini Transformer decoder: Embedding → Positional Encoding → Multi-Head Attention → Feed Forward
- Causal attention masking (no looking at future tokens)
- Next token prediction task (language modeling)
- DataParallel setup: `model = nn.DataParallel(model)` for large model
- Device management: move text tensors to appropriate device
- Multi-GPU batch size handling for long sequences
- Proper model saving/loading for multi-GPU Transformer
- Performance monitoring: tokens per second, perplexity
- Memory optimization: gradient checkpointing for long sequences
- **Time limit: 30 minutes**
- **Production-ready Transformer training**

**Self-Assessment Checklist:**
After each challenge, check if you can do these FROM MEMORY:

**Basic Skills:**
- [ ] Write nn.Module class structure without thinking
- [ ] Remember exact order of training loop (5 steps)
- [ ] Know common layer syntax by heart (Linear, Conv2d, etc.)
- [ ] Understand tensor shape transformations intuitively
- [ ] Setup optimizer và loss function correctly

**Intermediate Skills:**
- [ ] Handle train/validation splits properly
- [ ] Implement model saving/loading
- [ ] Calculate CNN output dimensions correctly
- [ ] Use BatchNorm và Dropout appropriately
- [ ] Setup proper data loading pipeline

**Advanced Skills:**
- [ ] Implement learning rate scheduling
- [ ] Add gradient clipping
- [ ] Handle device management (CPU/GPU)
- [ ] Create custom loss functions
- [ ] Implement early stopping logic

**Speed Benchmarks:**
- **Beginner**: Complete Challenge 1 in 10 minutes
- **Intermediate**: Complete Challenges 1-2 in 25 minutes
- **Advanced**: Complete Challenges 1-3 in 45 minutes
- **Expert**: Complete all challenges in 110 minutes

**Practice Progression:**
1. **Week 1**: Focus on Challenge 1, reduce time from 20min → 10min
2. **Week 2**: Add Challenge 2, aim for 15min completion
3. **Week 3**: Combine Challenges 1-2, complete in 25min total
4. **Week 4**: Add Challenge 3, master complete workflow
5. **Week 5**: Advanced challenges, production-ready code
6. **Week 6**: Final speed test, all challenges under time limits

**CRITICAL SUCCESS METRICS:**
- ✅ Can write basic neural network from scratch in 10 minutes
- ✅ Can implement text classifier with LSTM correctly in 15 minutes  
- ✅ Can create complete sentiment analysis pipeline in 20 minutes
- ✅ Can implement NER system with proper sequence labeling in 25 minutes
- ✅ Can build mini Transformer for language modeling in 30 minutes
- ✅ Code runs without syntax errors on first try
- ✅ All sequence tensor shapes match expectations (batch_first=True)
- ✅ Training loop produces decreasing loss và increasing accuracy
- ✅ Proper handling of variable-length sequences với padding
- ✅ Correct implementation of attention masking


```

#### Deliverables
- **Complete project implementation** (chosen option)
- **Technical documentation** explaining all design decisions
- **Performance analysis** with comprehensive visualizations
- **Code review checklist** demonstrating PyTorch best practices

#### Evaluation Criteria
- [ ] Demonstrates mastery of all PyTorch fundamentals
- [ ] Implements complex functionality correctly
- [ ] Code is production-ready quality
- [ ] Includes comprehensive testing and validation
- [ ] Shows creativity and advanced problem-solving skills

---

### Exercise 6.3: Syntax Reference Creation
**Type**: Summary & Review  
**Time**: 3-4 hours  
**Difficulty**: Beginner

#### Objective
Create personal reference sheet of all memorized syntax.

#### Requirements
Create comprehensive syntax reference from memory:

```python
# Personal PyTorch Syntax Reference
# (Write this from memory as final test)

# ===== TENSOR CREATION =====
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
rand = torch.rand(3, 4)
randn = torch.randn(3, 4)
arange = torch.arange(10)
linspace = torch.linspace(0, 1, 5)
tensor = torch.tensor([1, 2, 3])

# ===== TENSOR OPERATIONS =====
# Math operations
result = a + b
result = torch.add(a, b)
result = a * b
result = torch.matmul(a, b)

# Shape operations
reshaped = x.view(-1, 4)
reshaped = x.reshape(3, 4)
squeezed = x.squeeze()
unsqueezed = x.unsqueeze(0)
transposed = x.transpose(0, 1)
permuted = x.permute(2, 0, 1)

# ===== AUTOGRAD =====
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
x.grad.zero_()

# ===== NN.MODULE =====
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# ===== TRAINING LOOP =====
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# ===== COMMON LAYERS =====
# Add all layer syntax you memorized...
```

#### Deliverables
- **pytorch_syntax_reference.py**: Complete syntax reference from memory
- **daily_practice_routine.py**: Code patterns for continued practice
- **phase1_final_test.py**: Comprehensive self-assessment test

---

## Phase 1 Completion Assessment

### Final Mastery Test
Complete the following without any references:

**Syntax Fluency Test (30 minutes)**
1. Create a 3-layer neural network with batch normalization and dropout
2. Implement custom training loop with validation and early stopping
3. Add gradient clipping and learning rate scheduling
4. Write data loading pipeline for custom dataset

**Application Challenge (60 minutes)**
1. Build neural network for specific problem domain
2. Implement custom loss function appropriate for the problem
3. Add interpretability features to understand model decisions
4. Create comprehensive evaluation framework

**Integration Test (90 minutes)**
Choose and implement one complete project demonstrating:
- All fundamental PyTorch concepts
- Advanced training techniques
- Proper software engineering practices
- Creative problem-solving approaches

### Success Criteria
- **Syntax Mastery**: Write PyTorch code fluently without any references
- **Conceptual Understanding**: Apply concepts to solve novel, unseen problems
- **Integration Skills**: Combine multiple concepts into working systems
- **Code Quality**: Professional-level implementation and documentation
- **Problem Solving**: Demonstrate creativity and critical thinking

### Phase 1 Completion Requirements
- [ ] Complete all 6 weeks of exercises
- [ ] Pass all speed coding challenges within time limits
- [ ] Successfully implement final capstone project
- [ ] Create comprehensive personal syntax reference
- [ ] Demonstrate fluent PyTorch usage in timed assessments

**Upon successful completion**: Ready for **Phase 2 - Deep Learning Specialization** with focus on advanced architectures, CNN/RNN mastery, and domain-specific applications.

---

**Next Phase Preview**: Phase 2 will build on this foundation with advanced architectures (CNNs, RNNs, attention mechanisms), computer vision and NLP applications, and more complex thinking challenges while maintaining the proven learn-by-doing methodology.
