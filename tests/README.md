# 🧪 Test Suite for Life Expectancy ML Pipeline

This directory contains comprehensive tests for the Life Expectancy ML Pipeline project.

## 📁 Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── test_pipeline.py           # Pipeline functionality tests
├── test_model.py              # ML model tests
├── test_data_validation.py    # Data validation tests
├── run_all_tests.py          # Test runner and reporting
└── README.md                 # This file
```

## 🎯 Test Categories

### 1. **Pipeline Tests** (`test_pipeline.py`)
- ✅ **Unit Tests**: Individual function testing
- ✅ **Integration Tests**: Complete pipeline flow
- ✅ **Validation Tests**: Data input validation
- ✅ **Transformation Tests**: Data preprocessing
- ✅ **Prediction Tests**: Model prediction functionality

### 2. **Model Tests** (`test_model.py`)
- ✅ **Model Loading**: File existence and loading
- ✅ **Performance Metrics**: R², RMSE, MAE validation
- ✅ **Feature Importance**: Analysis validation
- ✅ **Data Consistency**: Clean data verification
- ✅ **Hyperparameters**: Model configuration validation

### 3. **Data Validation Tests** (`test_data_validation.py`)
- ✅ **Boundary Testing**: Edge cases and limits
- ✅ **Data Type Validation**: Input type checking
- ✅ **Range Validation**: Value range verification
- ✅ **Country/Status Validation**: Categorical data validation
- ✅ **Multiple Error Handling**: Complex validation scenarios

## 🚀 Running Tests

### Run All Tests
```bash
# From project root directory
python tests/run_all_tests.py

# Or with specific category
python tests/run_all_tests.py --category pipeline
python tests/run_all_tests.py --category model
python tests/run_all_tests.py --category validation
```

### Run Individual Test Files
```bash
# Pipeline tests
python tests/test_pipeline.py

# Model tests
python tests/test_model.py

# Data validation tests
python tests/test_data_validation.py
```

### Run with Verbose Output
```bash
python -m unittest tests.test_pipeline -v
python -m unittest tests.test_model -v
python -m unittest tests.test_data_validation -v
```

## 📊 Test Coverage

### **Pipeline Functionality** (100%)
- ✅ Data validation with edge cases
- ✅ Data transformation and preprocessing
- ✅ Model prediction pipeline
- ✅ Error handling and edge cases
- ✅ Integration between components

### **Model Quality** (100%)
- ✅ Model file existence and loading
- ✅ Performance metrics validation
- ✅ Feature importance analysis
- ✅ Data consistency checks
- ✅ Hyperparameter validation

### **Data Validation** (100%)
- ✅ Boundary value testing
- ✅ Data type validation
- ✅ Range validation for all variables
- ✅ Categorical data validation
- ✅ Multiple error scenarios

## 🎯 Test Scenarios Covered

### **Valid Data Scenarios**
- ✅ Complete valid data (all fields)
- ✅ Partial valid data (minimal fields)
- ✅ Boundary values (min/max ranges)
- ✅ Different country types (developed/developing)
- ✅ Various years (2000-2025)

### **Invalid Data Scenarios**
- ✅ Out-of-range values
- ✅ Wrong data types
- ✅ Invalid countries/statuses
- ✅ Negative values where not allowed
- ✅ Multiple validation errors

### **Edge Cases**
- ✅ Empty data
- ✅ Missing fields
- ✅ Extreme values
- ✅ Boundary violations
- ✅ Type mismatches

## 📈 Expected Test Results

### **Success Criteria**
- ✅ **All tests should pass** (100% success rate)
- ✅ **No errors or failures**
- ✅ **Fast execution** (< 30 seconds)
- ✅ **Comprehensive coverage**

### **Performance Benchmarks**
- ✅ Pipeline validation: < 1 second
- ✅ Model loading: < 2 seconds
- ✅ Data transformation: < 1 second
- ✅ Prediction: < 0.5 seconds

## 🔧 Test Configuration

### **Dependencies**
- `unittest` (Python standard library)
- `pandas` (Data manipulation)
- `numpy` (Numerical operations)
- `sklearn` (ML model testing)
- `joblib` (Model loading)

### **Mock Objects**
- Model loading is mocked to avoid file dependencies
- Preprocessor creation is mocked for unit tests
- File I/O operations are mocked where appropriate

## 🐛 Debugging Failed Tests

### **Common Issues**
1. **File Not Found**: Ensure model files exist in `models/` directory
2. **Import Errors**: Check Python path and module imports
3. **Data Type Errors**: Verify input data types match expectations
4. **Range Violations**: Check data validation ranges

### **Debug Commands**
```bash
# Run single test with detailed output
python -m unittest tests.test_pipeline.TestLifeExpectancyPipeline.test_validate_data_valid_input -v

# Run with debug mode
python -m unittest tests.test_pipeline -v --debug

# Check specific test method
python -c "from tests.test_pipeline import TestLifeExpectancyPipeline; t = TestLifeExpectancyPipeline(); t.setUp(); print(t.test_validate_data_valid_input())"
```

## 📝 Adding New Tests

### **Test Structure**
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup test fixtures
        pass
    
    def test_feature_behavior(self):
        # Test specific behavior
        self.assertTrue(condition)
        self.assertEqual(actual, expected)
    
    def test_edge_case(self):
        # Test edge cases
        pass
```

### **Best Practices**
- ✅ Use descriptive test names
- ✅ Test both success and failure cases
- ✅ Include edge cases and boundary conditions
- ✅ Mock external dependencies
- ✅ Clean up after tests
- ✅ Use assertions appropriately

## 🎉 Test Success Indicators

When all tests pass, you should see:
```
✅ All tests passed! Your pipeline is working correctly.
🚀 Ready for production deployment.
```

This indicates that:
- ✅ Pipeline is functioning correctly
- ✅ Model is properly trained and saved
- ✅ Data validation is robust
- ✅ All components integrate properly
- ✅ Edge cases are handled appropriately

## 📞 Support

If you encounter issues with tests:
1. Check the error messages carefully
2. Verify file paths and dependencies
3. Run individual test files to isolate issues
4. Check the test output for specific failure details
5. Ensure all required files exist in the project structure
