# MPSlib Hard and Soft Data Explanation

## 📋 **Overview**

This notebook demonstrates how MPSlib handles **hard data** (certain/known values) and **soft data** (uncertain/probabilistic values) in Multiple-Point Statistics simulations. It shows different approaches for incorporating both types of conditional data and how they affect the simulation results.

## 🔍 **Key Concepts**

### **Hard Data**
- **Definition**: Model parameters with **no uncertainty** (known values)
- **Format**: `[X, Y, Z, VALUE]` coordinates and exact value
- **Example**: A well log measurement at a specific location

### **Soft Data**
- **Definition**: Model parameters with **uncertainty** (probabilistic values)
- **Format**: `[X, Y, Z, P(category1), P(category2), ...]` coordinates and probabilities
- **Example**: Seismic interpretation with confidence levels

### **Co-located vs Non-co-located**
- **Co-located**: Soft data at the exact location being simulated
- **Non-co-located**: Soft data at different locations that influence the simulation

## 🎯 **Notebook Structure**

### **1. Setup and Configuration**

```python
# Uses mps_genesim algorithm (supports both co-located and non-co-located soft data)
O = mps.mpslib(method='mps_genesim')

# Training image with binary categories [0,1]
TI1, TI_filename1 = mps.trainingimages.strebelle(3, coarse3d=1)
O.par['soft_data_categories'] = np.array([0,1])
```

**Key Parameters:**
- `n_cond = 16`: Number of hard conditioning points
- `n_cond_soft = 1`: Number of soft conditioning points
- `n_real = 500`: Number of realizations
- `simulation_grid_size = [18, 13, 1]`: 2D grid

### **2. Data Definition**

#### **Hard Data (Optional)**
```python
d_hard = np.array([[15, 4, 0, 1],  # Location (15,4,0) with value 1
                   [15, 5, 0, 1]]) # Location (15,5,0) with value 1
# O.d_hard = d_hard  # Commented out - not used in this example
```

#### **Soft Data (Used)**
```python
d_soft = np.array([[ 2,  2, 0, 0.7, 0.3],    # Location (2,2,0): P(0)=0.7, P(1)=0.3
                   [ 5,  5, 0, 0.001, 0.999], # Location (5,5,0): P(0)=0.001, P(1)=0.999
                   [10,  8, 0, 0.999, 0.001]]) # Location (10,8,0): P(0)=0.999, P(1)=0.001
```

**Interpretation:**
- **Point 1**: Strong preference for category 0 (70% probability)
- **Point 2**: Strong preference for category 1 (99.9% probability)
- **Point 3**: Strong preference for category 0 (99.9% probability)

## 🧪 **Three Examples Demonstrated**

### **Example 1: Co-located Soft Data Only**

**Configuration:**
```python
O.par['n_cond_soft'] = 1
O.par['max_search_radius_soft'] = 0  # Only co-located data
```

**What happens:**
- Only uses soft data when simulating at the exact location of the soft data point
- If simulating at location (2,2,0), uses P(0)=0.7, P(1)=0.3
- If simulating elsewhere, ignores soft data

**Results:**
- **Unilateral path**: Sequential simulation in a fixed order
- **Random path**: Random order of simulation
- **Preferential path**: Simulate most informed locations first

### **Example 2: One Non-co-located Soft Data**

**Configuration:**
```python
O.par['n_cond_soft'] = 1
O.par['max_search_radius_soft'] = 1000000  # Large radius for non-co-located
```

**What happens:**
- Uses one soft data point from anywhere in the grid
- Algorithm searches for the most informative soft data point
- Can influence simulation at any location

### **Example 3: Three Non-co-located Soft Data**

**Configuration:**
```python
O.par['n_cond_soft'] = 3
O.par['max_search_radius_soft'] = 1000000  # Large radius
```

**What happens:**
- Uses all three soft data points simultaneously
- Maximum information from all available soft data
- Most computationally intensive but most informed

## 📊 **Visualization and Analysis**

### **Etype Statistics**
Each example generates:
- **Mean (m_mean)**: Average value across all realizations
- **Standard Deviation (m_std)**: Uncertainty measure
- **Mode (m_mode)**: Most frequent value

### **Plotting Results**
```python
plt.imshow(m_mean.T, vmin=0, vmax=1, cmap='hot')  # Mean values
plt.imshow(m_std.T, vmin=0, vmax=0.4, cmap='gray') # Uncertainty
```

## 🔄 **Simulation Paths**

### **1. Unilateral Path (`shuffle_simulation_grid = 0`)**
- Fixed sequential order (e.g., row by row)
- Fastest but may not use information optimally
- Good for simple cases

### **2. Random Path (`shuffle_simulation_grid = 1`)**
- Random order of simulation
- Balanced approach
- Moderate computational cost

### **3. Preferential Path (`shuffle_simulation_grid = 2`)**
- Simulates most informed locations first
- Uses entropy to determine order
- Most computationally efficient for soft data
- **Recommended** when using soft data

## 🎯 **Key Insights**

### **Impact of Soft Data**
1. **Co-located**: Only affects simulation at exact locations
2. **Non-co-located**: Influences entire simulation domain
3. **Multiple soft data**: Provides maximum information but increases computation

### **Path Selection**
- **Preferential path** is most efficient with soft data
- **Random path** can be sufficient for simple cases
- **Unilateral path** is fastest but least informed

### **Algorithm Choice**
- **`mps_genesim`**: Supports both co-located and non-co-located soft data
- **`mps_snesim_tree`**: Only co-located soft data
- **`mps_snesim_list`**: Only co-located soft data

## 📈 **Practical Applications**

### **When to Use Hard Data**
- Well log measurements
- Core sample data
- Known geological boundaries

### **When to Use Soft Data**
- Seismic interpretations
- Geophysical surveys
- Expert knowledge with uncertainty

### **When to Use Co-located vs Non-co-located**
- **Co-located**: When you have direct measurements with uncertainty
- **Non-co-located**: When you have regional information that should influence the entire domain

## 🔧 **Parameter Tuning**

### **For Better Results**
1. **Use preferential path** (`shuffle_simulation_grid = 2`)
2. **Increase `n_cond_soft`** for more information
3. **Set appropriate `max_search_radius_soft`** based on data spacing
4. **Use `mps_genesim`** for non-co-located soft data

### **For Faster Computation**
1. **Reduce `n_cond_soft`** to minimum necessary
2. **Use co-located data** when possible
3. **Limit `max_search_radius_soft`** for large datasets

## 🎓 **Learning Outcomes**

This notebook teaches:
1. **How to incorporate different types of conditional data**
2. **Impact of simulation path on results**
3. **Trade-offs between information and computation**
4. **Visualization of uncertainty in MPS simulations**
5. **Best practices for parameter selection**

The examples demonstrate that **soft data can significantly improve simulation quality** when properly incorporated, and that **preferential simulation paths** are essential for efficient use of uncertain information.

## 🔍 **How `n_cond_soft` Impacts the Simulation**

### **Understanding the Difference: `n_cond` vs `n_cond_soft`**

#### **`n_cond` (Hard Data)**
- **What it does**: Controls how many **hard data points** are used to condition each simulation node
- **Probability tree**: Each hard data point provides **100% certainty** about a specific value
- **Impact**: Reduces uncertainty by fixing known values in the simulation

#### **`n_cond_soft` (Soft Data)**
- **What it does**: Controls how many **soft data points** are used to condition each simulation node
- **Probability tree**: Each soft data point provides **probabilistic information** that modifies the conditional probability distribution
- **Impact**: **Shifts and reshapes** the probability distribution rather than fixing values

### **The Probability Tree with Soft Data**

When `n_cond_soft = 1`:
```
Original TI Pattern Probability: P(pattern) = 0.3
Soft Data at location: P(value=1) = 0.8

Modified Probability: P(pattern | soft_data) = P(pattern) × P(soft_data | pattern)
```

When `n_cond_soft = 3`:
```
Original TI Pattern Probability: P(pattern) = 0.3
Soft Data 1: P(value=1) = 0.8
Soft Data 2: P(value=0) = 0.9  
Soft Data 3: P(value=1) = 0.2

Modified Probability: P(pattern | all_soft_data) = P(pattern) × P(soft_data1 | pattern) × P(soft_data2 | pattern) × P(soft_data3 | pattern)
```

### **Step-by-Step Impact Process**

#### **1. Pattern Matching Phase**
```python
# Algorithm finds matching patterns in training image
matching_patterns = find_patterns_in_TI(template, hard_data, soft_data)
```

#### **2. Soft Data Conditioning**
```python
# For each matching pattern, calculate modified probability
for pattern in matching_patterns:
    original_prob = pattern.frequency_in_TI
    
    # Apply soft data conditioning
    modified_prob = original_prob
    for soft_point in soft_data_points:
        if soft_point.location in pattern:
            # Multiply by soft data probability
            soft_prob = soft_point.probabilities[pattern.value_at_location]
            modified_prob *= soft_prob
    
    pattern.final_probability = modified_prob
```

#### **3. Sampling from Modified Distribution**
```python
# Sample from the modified probability distribution
sampled_value = sample_from_distribution(modified_probabilities)
```

### **Visual Example: How `n_cond_soft` Changes Results**

#### **Case 1: `n_cond_soft = 0` (No Soft Data)**
```
Training Image Pattern: [0,1,0] → P = 0.4
Result: Pattern appears in 40% of realizations
```

#### **Case 2: `n_cond_soft = 1` (One Soft Data Point)**
```
Training Image Pattern: [0,1,0] → P = 0.4
Soft Data at position 1: P(1) = 0.8
Modified Probability: 0.4 × 0.8 = 0.32
Result: Pattern appears in 32% of realizations (reduced due to soft data preference)
```

#### **Case 3: `n_cond_soft = 3` (Three Soft Data Points)**
```
Training Image Pattern: [0,1,0] → P = 0.4
Soft Data 1 at pos 1: P(1) = 0.8
Soft Data 2 at pos 2: P(0) = 0.9  # Conflicts with pattern!
Soft Data 3 at pos 3: P(0) = 0.7
Modified Probability: 0.4 × 0.8 × 0.1 × 0.7 = 0.0224
Result: Pattern appears in only 2.24% of realizations (strongly reduced due to conflicts)
```

### **Key Differences from `n_cond`**

| Aspect | `n_cond` (Hard Data) | `n_cond_soft` (Soft Data) |
|--------|---------------------|---------------------------|
| **Type of Information** | Certain values | Probabilistic preferences |
| **Impact on Patterns** | Filters out incompatible patterns | Modifies pattern probabilities |
| **Computational Cost** | Low (simple filtering) | High (probability calculations) |
| **Uncertainty** | Reduces uncertainty | Manages uncertainty |
| **Flexibility** | Rigid constraints | Flexible influence |

### **Why `n_cond_soft` Matters**

#### **1. Information Integration**
- **More soft data points** = more information integrated into the simulation
- **Better informed** decisions about pattern selection
- **More realistic** geological models

#### **2. Uncertainty Quantification**
- **Soft data** preserves uncertainty rather than eliminating it
- **Multiple soft data points** can show conflicting information
- **Etype statistics** reflect this uncertainty in the results

#### **3. Geological Realism**
- **Real data** is often uncertain (seismic, geophysical)
- **Expert knowledge** can be incorporated as soft data
- **Regional trends** can influence local simulations

### **Practical Impact Examples**

#### **Example A: Channel Simulation**
```
Training Image: Shows channel patterns
Soft Data 1: Seismic suggests channel at location A (P=0.7)
Soft Data 2: Well data suggests no channel at location B (P=0.9)
Soft Data 3: Regional trend suggests channel direction (P=0.6)

n_cond_soft = 1: Uses only one piece of information
n_cond_soft = 3: Integrates all three pieces of information
```

#### **Example B: Facies Distribution**
```
Training Image: Shows facies transitions
Soft Data: Multiple seismic interpretations with different confidence levels

n_cond_soft = 1: Uses most confident interpretation only
n_cond_soft = 5: Uses all interpretations, weighted by confidence
```

### **Computational Considerations**

#### **Performance Impact**
```python
# Computational complexity increases with n_cond_soft
complexity ≈ O(n_patterns × n_cond_soft × n_categories)
```

#### **Memory Usage**
- **More soft data points** = more probability calculations
- **Larger search radius** = more potential soft data points to consider
- **Multiple categories** = more complex probability distributions

### **Best Practices for `n_cond_soft`**

#### **1. Start Small**
```python
# Begin with n_cond_soft = 1
O.par['n_cond_soft'] = 1
# Test results, then increase if needed
```

#### **2. Consider Data Quality**
```python
# High-quality soft data: use more points
if soft_data_quality == 'high':
    O.par['n_cond_soft'] = 3
else:
    O.par['n_cond_soft'] = 1
```

#### **3. Balance Information vs Computation**
```python
# Trade-off: more information vs faster computation
if computation_time_available == 'high':
    O.par['n_cond_soft'] = 5
else:
    O.par['n_cond_soft'] = 2
```

### **Debugging `n_cond_soft` Issues**

#### **Common Problems**
1. **Too many soft data points**: Can overwhelm the training image patterns
2. **Conflicting soft data**: Can lead to very low probabilities
3. **Poor soft data quality**: Can introduce noise rather than information

#### **Diagnostic Tools**
```python
# Check soft data impact
O.par['debug_level'] = 1  # Enable debugging
# Look for probability modifications in output
```

### **Summary: The Power of Soft Data**

**`n_cond_soft`** transforms MPSlib from a **pattern-matching algorithm** into a **probabilistic integration system**:

- **`n_cond = 0`**: Pure pattern simulation (no conditioning)
- **`n_cond > 0`**: Pattern simulation with hard constraints
- **`n_cond_soft > 0`**: Pattern simulation with **probabilistic guidance**

The key insight is that **soft data doesn't just filter patterns** (like hard data does) - it **modifies the probability of selecting each pattern** based on how well that pattern aligns with the uncertain information you have about the subsurface.
