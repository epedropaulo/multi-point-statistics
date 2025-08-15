# Simple Usage Guide

## 🎯 **Two Simple Functions**

### **1. Save Function: `save_binary_as_ti_and_npy()`**
Replaces `load_binary_from_eleven_sandstones` and saves in both formats.

### **2. Load Function: `load_ti_from_npy()`**
Loads training images from `.npy` files.

## 📝 **How to Use in Your Notebook**

### **Step 1: Import**
```python
from scripts.ti_saver import save_binary_as_ti_and_npy, load_ti_from_npy
```

### **Step 2: Save Your Data**
```python
# Instead of using load_binary_from_eleven_sandstones
# Use this to save in both .dat and .npy formats:

binary_data = your_binary_data  # Your data array
output_path = "../data/eas/Parker_binary_0"  # No extension needed

dat_path, npy_path = save_binary_as_ti_and_npy(binary_data, output_path)
```

### **Step 3: Load Later**
```python
# Load the .npy file
ti_array = load_ti_from_npy("../data/eas/Parker_binary_0.npy")

# Use in MPSlib
O = mps.mpslib()
O.ti = ti_array
```

## 📊 **Example Output**

When saving:
```
✅ Saved as .dat: ../data/eas/Parker_binary_0.dat
✅ Saved as .npy: ../data/eas/Parker_binary_0.npy
   Shape: (1000, 1000, 1), Type: float64
```

When loading:
```
✅ Loaded TI from: ../data/eas/Parker_binary_0.npy
   Shape: (1000, 1000, 1), Type: float64
```

## 🎉 **That's It!**

- **One function** to save in both formats
- **One function** to load .npy files
- **Simple and clean**
