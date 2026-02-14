# Agentic AI for Image Classification

A multi-agent system that uses Apache Kafka as a message broker to compare two deep learning models (CNN and ResNet) classifying CIFAR-10 images. A Judge agent orchestrates the comparison and selects the best prediction based on confidence scores.

## Architecture

**Agents:**

- **CNN-CIFAR** -- Custom convolutional neural network with batch normalization and dropout
- **RESNET-CIFAR** -- ResNet-style model with residual (skip) connections
- **Judge** -- Sends test images to both agents via Kafka, collects responses, picks the higher-confidence prediction as the "brokered" result, and visualizes accuracy

## Diagrams

### Notebook Dependency Graph

Shows the dependency chain between the notebook steps -- what must complete before each subsequent step can run.

```mermaid
%%{init: {'theme': 'forest'}}%%
graph TD
    S1["Step 1: Install Packages<br/>& Start Kafka"]
    S2["Step 2: Load CIFAR-10<br/>& Helper Functions"]
    S3["Step 3: Build & Train<br/>CNN Model"]
    S4["Step 4: Build & Train<br/>ResNet Model"]
    S5["Step 5: CNN Agent<br/>(Consumer/Producer)"]
    S6["Step 6: ResNet Agent<br/>(Consumer/Producer)"]
    S7["Step 7: Judge Agent<br/>Definition"]
    S8["Step 8: Orchestration<br/>& Threading"]
    S9["Step 9: Restart Kafka"]
    S10["Step 10: Execute &<br/>Visualize Results"]

    S1 --> S2
    S2 --> S3
    S2 --> S4
    S3 --> S5
    S4 --> S6
    S2 --> S7
    S5 --> S8
    S6 --> S8
    S7 --> S8
    S8 --> S9
    S9 --> S10
```

### Software Dependency Stack

External libraries and services required by the system.

```mermaid
%%{init: {'theme': 'forest'}}%%
graph BT
    subgraph Infrastructure
        JAVA["Java 17+ (JRE)"]
        ZK["Apache Zookeeper"]
        KAFKA["Apache Kafka 3.5.0"]
        JAVA --> ZK --> KAFKA
    end

    subgraph Python Libraries
        TF["TensorFlow / Keras"]
        NP["NumPy"]
        MPL["Matplotlib"]
        PIL["Pillow"]
        KP["kafka-python"]
        TQDM["tqdm"]
    end

    subgraph Models
        CNN["CNN Model"]
        RESNET["ResNet Model"]
    end

    subgraph Agents
        CNN_A["CNN Agent"]
        RES_A["ResNet Agent"]
        JUDGE["Judge Agent"]
    end

    TF --> CNN
    TF --> RESNET
    NP --> CNN
    NP --> RESNET
    CNN --> CNN_A
    RESNET --> RES_A
    KP --> CNN_A
    KP --> RES_A
    KP --> JUDGE
    KAFKA --> CNN_A
    KAFKA --> RES_A
    KAFKA --> JUDGE
    MPL --> JUDGE
    NP --> JUDGE
```

### Process Flow

End-to-end execution sequence from setup through results.

```mermaid
%%{init: {'theme': 'forest'}}%%
sequenceDiagram
    participant U as User / Notebook
    participant K as Kafka Broker
    participant C as CNN Agent (Thread)
    participant R as ResNet Agent (Thread)
    participant J as Judge Agent (Thread)

    Note over U: Step 1-2: Setup & load data
    Note over U: Step 3: Train CNN model
    Note over U: Step 4: Train ResNet model

    U->>C: Start CNN agent thread
    activate C
    U->>R: Start ResNet agent thread
    activate R

    Note over U: Step 8: Start orchestration

    U->>J: Start Judge agent thread
    activate J

    loop For each of 500 test images
        J->>K: Publish image request
        K->>C: Deliver request (consumer group)
        K->>R: Deliver request (consumer group)
        C->>C: Classify with CNN
        R->>R: Classify with ResNet
        C->>K: Publish CNN result
        R->>K: Publish ResNet result
    end

    K->>J: Deliver CNN results
    K->>J: Deliver ResNet results
    J->>J: Compare & pick best

    deactivate C
    deactivate R

    J->>U: Return accuracy stats
    deactivate J

    Note over U: Step 10: Visualize bar chart
```

### Data Flow

How image data moves through the system and transforms at each stage.

```mermaid
%%{init: {'theme': 'forest'}}%%
flowchart LR
    subgraph Input
        CIFAR[("CIFAR-10<br/>Dataset<br/>60k images")]
    end

    subgraph Preprocessing
        LOAD["Load & Split<br/>Train / Test"]
        NORM["Normalize<br/>0-255 → 0-1"]
        CIFAR --> LOAD --> NORM
    end

    subgraph Training
        direction TB
        CNN_T["Train CNN<br/>(40k images)"]
        RES_T["Train ResNet<br/>(40k images)"]
        VAL["Validation<br/>(10k images)"]
    end

    NORM --> CNN_T
    NORM --> RES_T
    NORM --> VAL

    subgraph Kafka Topics
        REQ[/"cifar_classification_requests<br/>(image JSON)"/]
        CNN_R[/"cnn_classifications<br/>(prediction JSON)"/]
        RES_R[/"resnet_classifications<br/>(prediction JSON)"/]
    end

    subgraph Agent Processing
        JUDGE_PUB["Judge picks<br/>random test image"]
        CNN_INF["CNN inference<br/>→ softmax predictions"]
        RES_INF["ResNet inference<br/>→ softmax predictions"]
    end

    NORM --> JUDGE_PUB
    JUDGE_PUB --> REQ
    REQ --> CNN_INF
    REQ --> RES_INF
    CNN_INF --> CNN_R
    RES_INF --> RES_R

    subgraph Output
        COMPARE["Compare predictions<br/>by confidence"]
        CHART["Accuracy<br/>bar chart"]
    end

    CNN_R --> COMPARE
    RES_R --> COMPARE
    COMPARE --> CHART
```

### Kafka Topic Message Schemas

Structure of the JSON messages passed through Kafka.

```mermaid
%%{init: {'theme': 'forest'}}%%
classDiagram
    class ClassificationRequest {
        int image_id
        float[][][] image_data  (32x32x3)
        int true_label
        string true_class
    }

    class ClassificationResponse {
        string agent
        int image_id
        int true_label
        string true_class
        int predicted_class
        string predicted_name
        float confidence
        float inference_time_ms
        string timestamp
    }

    class JudgeResult {
        int total
        int cnn_correct
        int resnet_correct
        int brokered_correct
    }

    ClassificationRequest --> ClassificationResponse : CNN or ResNet processes
    ClassificationResponse --> JudgeResult : Judge aggregates
```

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~krig/cifar.html) -- 60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Requirements

- Python 3.10+
- TensorFlow / Keras
- Apache Kafka 3.5.0 (requires Java 17+)
- kafka-python
- numpy, matplotlib, pillow, tqdm

## Quick Start

The notebook was originally built for Google Colab. To run on a local or VPS environment, see [RUN_ON_VPS.md](RUN_ON_VPS.md) for full setup instructions covering both CPU-only and GPU configurations.

## Files

| File | Description |
|---|---|
| `image_class.ipynb` | Main notebook -- trains models, runs agents, visualizes results |
| `RUN_ON_VPS.md` | Deployment guide for Ubuntu 24.04 (CPU and GPU) |

## Status

Work in progress -- actively improving and expanding the system.
