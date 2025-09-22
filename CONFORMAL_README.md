# Conformal Prediction MapReduce

A Hadoop MapReduce implementation for Conformal Prediction on the FireSmoke dataset, similar to the classic WordCount example but designed for machine learning confidence estimation.

## Overview

This project implements conformal prediction using Hadoop MapReduce to process logits data from CLIP models and compute prediction sets with statistical coverage guarantees. The implementation follows the same structure as WordCount but processes numerical data instead of text.

## What is Conformal Prediction?

Conformal prediction is a framework that provides prediction sets (rather than point predictions) with statistical coverage guarantees. For a given confidence level (e.g., 90%), the prediction set is guaranteed to contain the true label with at least that probability.

## Project Structure

```
data-clip-conformo/
├── ConformalPrediction.java           # Main driver class
├── ConformalPredictionMapper.java     # Mapper implementation
├── ConformalPredictionReducer.java    # Reducer implementation
├── compile_conformal_prediction.bat   # Compilation script
├── run_conformal_prediction.bat       # Execution script
├── generate_sample_data.py            # Data generation script
├── input_data/                        # Sample input data
│   ├── conformal_data.txt
│   └── (generated files)
├── ConformalPrediction/               # Compiled classes directory
└── CONFORMAL_README.md                # This file
```

## Prerequisites

1. **Hadoop Installation**: Hadoop 3.x installed and configured
2. **Java**: JDK 8 or higher
3. **Environment Variables**: `HADOOP_HOME` properly set
4. **Python**: For data generation (optional)

## Input Data Format

The input data should be in CSV format with the following structure:
```
image_path,label,logit1,logit2,logit3,logit4
```

Where:
- `image_path`: Path to the image file
- `label`: True class label (0-3)
- `logit1-logit4`: Raw model outputs (logits) for each class

Classes:
- 0: bothFireAndSmoke
- 1: fire  
- 2: smoke
- 3: neitherFireNorSmoke

## Step-by-Step Instructions

### 1. Generate Sample Data

First, generate sample input data:

```bash
python generate_sample_data.py
```

This creates sample logits data in the `input_data/` directory.

### 2. Compile the MapReduce Code

Navigate to the project directory and run:

```bash
compile_conformal_prediction.bat
```

This script:
- Compiles all Java files
- Creates the `ConformalPrediction.jar` file
- Shows next steps

**Manual compilation (if script fails):**
```bash
# Create directory
mkdir ConformalPrediction

# Compile Java files
javac -classpath "%HADOOP_HOME%\share\hadoop\common\*";"%HADOOP_HOME%\share\hadoop\mapreduce\*" -d ConformalPrediction\ ConformalPrediction.java ConformalPredictionMapper.java ConformalPredictionReducer.java

# Create JAR
jar -cvf ConformalPrediction.jar -C ConformalPrediction\ .
```

### 3. Start Hadoop Services

Make sure Hadoop services are running:

```bash
start-dfs
start-yarn
```

Verify services are running:
```bash
jps
```

You should see: NameNode, DataNode, ResourceManager, NodeManager

### 4. Prepare HDFS

Create input directory in HDFS:
```bash
hadoop fs -mkdir /conformal_input
```

Upload input data:
```bash
hadoop fs -put input_data/* /conformal_input
```

Verify data upload:
```bash
hadoop fs -ls /conformal_input
hadoop fs -head /conformal_input/conformal_data.txt
```

### 5. Run the MapReduce Job

Execute the conformal prediction job:

```bash
run_conformal_prediction.bat
```

**Manual execution:**
```bash
hadoop jar ConformalPrediction.jar ConformalPrediction /conformal_input /conformal_output
```

### 6. View Results

Check the output:
```bash
hadoop fs -cat /conformal_output/part-r-00000
hadoop fs -ls /conformal_output
```

Copy results to local filesystem:
```bash
hadoop fs -get /conformal_output/part-r-00000 conformal_results.txt
```

## Expected Output

The MapReduce job produces several metrics:

### Training Data Metrics
- `TRAIN_ACCURACY`: Classification accuracy on training data
- `TRAIN_TOTAL_SAMPLES`: Number of training samples processed
- `TRAIN_CLASS_*_ACCURACY`: Per-class accuracy
- `CONFORMAL_THRESHOLD`: Threshold for 90% coverage

### Test Data Metrics  
- `TEST_ACCURACY`: Classification accuracy on test data
- `TEST_COVERAGE`: Conformal prediction coverage
- `TEST_AVG_SET_SIZE`: Average prediction set size
- `TEST_TOTAL_SAMPLES`: Number of test samples processed

Example output:
```
CONFORMAL_THRESHOLD    0.876543
TEST_ACCURACY          0.8250
TEST_AVG_SET_SIZE      1.45
TEST_COVERAGE          0.9100
TRAIN_ACCURACY         0.8750
```

## Algorithm Details

### Mapper Phase
1. **Input Processing**: Parses CSV lines containing image paths, labels, and logits
2. **Softmax Calculation**: Converts logits to probabilities using numerically stable softmax
3. **Prediction**: Determines predicted class (argmax of probabilities)
4. **Non-conformity Score**: Calculates 1 - P(true_class) for conformal prediction
5. **Output**: Emits (dataset_type, prediction_data) pairs

### Reducer Phase
1. **Aggregation**: Collects all predictions for train/test splits
2. **Accuracy Calculation**: Computes classification accuracy
3. **Conformal Threshold**: Determines threshold for desired coverage (90%)
4. **Coverage Estimation**: Estimates coverage on test data
5. **Class Statistics**: Computes per-class metrics

## Comparison with Original Python Implementation

| Aspect | Python Version | MapReduce Version |
|--------|---------------|-------------------|
| **Scalability** | Single machine | Distributed cluster |
| **Data Size** | Limited by RAM | Limited by HDFS |
| **Processing** | Sequential | Parallel |
| **Fault Tolerance** | None | Built-in recovery |
| **Setup** | Simple | Complex |
| **Performance** | Fast for small data | Fast for large data |

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Check `HADOOP_HOME` is set correctly
   - Verify JDK version compatibility
   - Ensure Hadoop classpath is accessible

2. **HDFS Errors**
   - Check Hadoop services are running: `jps`
   - Verify HDFS is accessible: `hadoop fs -ls /`
   - Check permissions: `hadoop fs -chmod 755 /conformal_input`

3. **Job Failures**
   - Check logs: `yarn logs -applicationId <app_id>`
   - Verify input data format
   - Monitor resource usage

## References

- [Conformal Prediction: A Gentle Introduction](https://arxiv.org/abs/2107.07511)
- [Hadoop MapReduce Tutorial](https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)