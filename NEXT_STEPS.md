# Next Steps: Improving and Expanding the Image Classification System

## Section 1: Improving What the Notebook Already Does

These changes make the existing system more robust, accurate, and maintainable without changing its scope.

### 1.1 Fix the ResNet Data Augmentation Bug

The ResNet model defines a `RandomFlip` layer but then passes `inputs` (not `x`) to the first Conv2D, so the flip is never actually applied. This means the ResNet trains without data augmentation while the CNN gets it, making the comparison unfair.

```python
# Current (broken) -- the flip output is discarded:
x = layers.RandomFlip("horizontal")(inputs)
x = layers.Conv2D(64, (3, 3), padding='same')(inputs)  # <-- uses inputs, not x

# Fixed:
x = layers.RandomFlip("horizontal")(inputs)
x = layers.Conv2D(64, (3, 3), padding='same')(x)       # <-- uses x
```

**Value:** Gives the ResNet the same augmentation advantage the CNN already has, producing a fair comparison and likely improving ResNet accuracy by 2-5%.

### 1.2 Add Data Augmentation Beyond Horizontal Flip

Both models only use `RandomFlip("horizontal")`. CIFAR-10 benefits significantly from additional augmentation.

- `RandomRotation(0.05)` -- slight rotations
- `RandomZoom(0.1)` -- minor scale changes
- `RandomTranslation(0.1, 0.1)` -- small shifts

**Value:** Reduces overfitting and can improve test accuracy by 3-8% on CIFAR-10, especially for the CNN which has fewer parameters.

### 1.3 Use the Full Test Set for Evaluation

Both models are evaluated on only 1,000 of the 10,000 test images:

```python
cnn_model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
```

Change to `x_test, y_test` to use all 10,000. The evaluation pass is fast (no gradients computed) and gives a more statistically reliable accuracy number.

**Value:** More trustworthy accuracy reporting. Results on 1,000 samples can swing by 1-2% depending on which images happen to be in that slice.

### 1.4 Add Per-Class Metrics and a Confusion Matrix

The notebook only reports overall accuracy. Adding per-class precision, recall, and a confusion matrix reveals which classes each model struggles with (e.g., cat vs. dog, automobile vs. truck).

**Value:** Understanding failure modes is more useful than a single accuracy number. It reveals whether the models make the same mistakes or different ones, which directly justifies the multi-agent approach.

### 1.5 Add Training History Plots

The training histories (`history_cnn`, `history_resnet`) are captured but never visualized. Plotting loss and accuracy curves for both training and validation sets over epochs reveals overfitting, underfitting, and convergence behavior.

**Value:** Makes it easy to diagnose training problems and justify the early stopping and learning rate reduction callbacks. Essential for any model development workflow.

### 1.6 Improve the Judge's Selection Strategy

The Judge currently picks the agent with the higher confidence score. High confidence does not always mean correctness -- a poorly calibrated model can be confidently wrong.

Better strategies:
- **Weighted voting** -- weight each agent's vote by its historical accuracy on that class
- **Agreement check** -- if both agents agree, use that answer; if they disagree, apply a tiebreaker
- **Confidence thresholding** -- only trust predictions above a calibrated threshold

**Value:** The "pick the most confident" strategy can actually perform worse than either individual model if one model is overconfident. A smarter judge is the whole point of the multi-agent architecture.

### 1.7 Set a Random Seed for Reproducibility

The notebook uses `np.random.randint` without a seed, so every run tests different images, making results non-comparable across runs.

```python
np.random.seed(42)
tf.random.set_seed(42)
```

**Value:** Makes experiments repeatable and results comparable. Essential for validating that a change actually improved performance rather than just getting lucky with the random sample.

### 1.8 Save and Load Trained Models

The model save code is commented out and tied to Google Drive. Add local saving:

```python
cnn_model.save('cnn_cifar_model.keras')
resnet_model.save('resnet_cifar_model.keras')
```

And add a cell that loads them if they exist, skipping training entirely:

```python
if os.path.exists('cnn_cifar_model.keras'):
    cnn_model = keras.models.load_model('cnn_cifar_model.keras')
else:
    # train the model
```

**Value:** Avoids retraining on every notebook restart. On CPU this saves 30-60 minutes per run. On GPU it still saves several minutes and allows separating model development from agent testing.

### 1.9 Decouple Kafka Setup from the Notebook

Move Kafka installation and management out of the notebook and into external scripts or systemd services (as described in RUN_ON_VPS.md). Replace the inline `!curl`, `!tar`, and shell commands with a simple health check:

```python
from kafka import KafkaProducer
try:
    p = KafkaProducer(bootstrap_servers=['localhost:9092'])
    p.close()
    print("Kafka is reachable")
except Exception as e:
    raise RuntimeError("Kafka is not running. Start it before running this notebook.") from e
```

**Value:** Separates infrastructure from application logic. Eliminates the fragile Step 9 Kafka restart cell, reduces notebook complexity, and makes the notebook portable across environments.

### 1.10 Add Error Handling to Agents

The agents have no error handling. If Kafka is down, a message is malformed, or inference fails, the agent thread dies silently. Add try/except blocks with logging inside the message processing loops.

**Value:** Without error handling, a single bad message kills an agent thread with no feedback. In a real multi-agent system, resilience is a core requirement.

---

## Section 2: New Capabilities and Concepts

These enhancements extend the system beyond its current scope to demonstrate additional concepts in multi-agent AI, MLOps, and distributed systems.

### 2.1 Add a Third Model Agent (MobileNet or EfficientNet)

Add a lightweight pre-trained model like MobileNetV2 or EfficientNetB0 using transfer learning on CIFAR-10. This agent would subscribe to the same Kafka topic and publish to its own response topic.

**Value:** Demonstrates transfer learning (using a model pre-trained on ImageNet and fine-tuning it on CIFAR-10) and shows how the pub/sub architecture scales to N agents without code changes to existing agents. Transfer learning typically outperforms training from scratch on small datasets, so this agent would likely become the strongest classifier.

### 2.2 Implement a Weighted Ensemble Judge

Replace the simple "pick highest confidence" strategy with a true ensemble that combines predictions from all agents using learned weights or voting schemes:

- **Soft voting** -- average the softmax probability vectors from all agents, then pick the argmax
- **Stacking** -- train a small meta-model (logistic regression or small neural net) that takes all agents' predictions as input and learns the optimal combination
- **Dynamic weighting** -- weight each agent by its running accuracy on the current class

**Value:** Ensemble methods are a fundamental concept in machine learning. They almost always outperform individual models because different architectures make different errors. This demonstrates why the multi-agent approach is architecturally justified, not just a messaging exercise.

### 2.3 Add a Real-Time Performance Dashboard

Replace the static matplotlib bar chart with a live-updating dashboard using Streamlit or Plotly Dash that shows:

- Running accuracy for each agent
- Per-class accuracy heatmap
- Inference latency distribution
- Agent agreement rate
- Live message throughput

**Value:** Demonstrates real-time data visualization and monitoring, which is essential in production ML systems. Makes the multi-agent behavior visible as it happens rather than only as a post-hoc summary.

### 2.4 Support Custom Image Upload for Classification

Add the ability to upload an image (not from CIFAR-10) and have all agents classify it. This requires:

- An upload mechanism (file upload widget in Jupyter, or an HTTP endpoint)
- Image preprocessing to resize and normalize to 32x32x3
- Publishing the custom image through the same Kafka pipeline

**Value:** Demonstrates that the system works on real-world data, not just the dataset it was trained on. Exposes limitations of models trained on tiny 32x32 images and motivates moving to higher-resolution datasets and architectures.

### 2.5 Swap CIFAR-10 for a Higher-Resolution Dataset

Replace or supplement CIFAR-10 with a more challenging dataset like CIFAR-100 (100 classes), Tiny ImageNet (200 classes, 64x64), or a custom dataset. This requires adjusting:

- Input shapes in model definitions
- Number of output classes
- Data loading and preprocessing
- Class name mappings

**Value:** CIFAR-10 is a toy benchmark. Moving to harder datasets demonstrates scalability and exposes real differences in model architectures. ResNet's skip connections become more important as depth and difficulty increase.

### 2.6 Containerize with Docker Compose

Package the entire system into Docker containers:

- `kafka` + `zookeeper` containers (use Confluent or Bitnami images)
- `jupyter` container with TensorFlow and all dependencies
- Optional: separate containers for each agent (true microservice architecture)

**Value:** Demonstrates containerization and infrastructure-as-code. Makes the system reproducible on any machine with Docker. Eliminates the "it works on my machine" problem entirely. Also opens the door to deploying on Kubernetes.

### 2.7 Replace Kafka with a Lightweight Alternative

Kafka is powerful but heavy for this use case. Demonstrate the same architecture with a lighter message broker:

- **Redis Pub/Sub or Redis Streams** -- minimal setup, built-in with many hosting providers
- **RabbitMQ** -- purpose-built message broker with better routing semantics
- **ZeroMQ** -- brokerless, embedded in the Python process

**Value:** Shows that the agent architecture is broker-agnostic. Comparing brokers teaches the tradeoffs between throughput, persistence, ordering guarantees, and operational complexity. Redis Streams would eliminate the Java dependency entirely.

### 2.8 Add Model Versioning and A/B Testing

Save each trained model with a version tag and track performance across versions. The Judge agent can route traffic (e.g., 80% to the current best model, 20% to a new candidate) and compare their live performance.

**Value:** Demonstrates MLOps concepts: model registry, canary deployment, and A/B testing. These are standard practices in production ML systems and show how the multi-agent architecture naturally supports experimentation.

### 2.9 Add Agent Communication and Negotiation

Instead of the Judge making all decisions, allow agents to communicate with each other:

- If CNN and ResNet disagree, they exchange their confidence scores
- The lower-confidence agent can defer to the higher-confidence one
- Agents can request a "second opinion" on images they're uncertain about

**Value:** Demonstrates multi-agent negotiation and cooperative decision-making, which are core concepts in agentic AI. Moves the system from a simple fan-out/fan-in pattern to genuine agent interaction.

### 2.10 Export Results to CSV/JSON and Add Statistical Analysis

Save all classification results to structured files and perform statistical analysis:

- Per-class accuracy with confidence intervals
- McNemar's test to determine if the accuracy difference between models is statistically significant
- Inference time distributions and outlier analysis

**Value:** Raw accuracy numbers without statistical context can be misleading. A 1% accuracy difference on 500 samples may not be statistically significant. This adds rigor and demonstrates proper experimental methodology.

---

## Section 3: Implementation Guide

### 3.1 Fix the ResNet Bug and Add Augmentation (Sections 1.1, 1.2)

In the `build_resnet_model()` function (Step 4), fix the data flow and add augmentation layers:

```python
def build_resnet_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Data augmentation pipeline
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    # Initial Conv -- note: uses x, not inputs
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # ... rest of model unchanged
```

Apply the same augmentation layers to the CNN's `build_cnn_model()` function (Step 3):

```python
model = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    # ... rest unchanged
])
```

### 3.2 Add Confusion Matrix and Training Plots (Sections 1.4, 1.5)

Add a new cell after each model's training cell:

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# After CNN training (Step 3)
y_pred_cnn = np.argmax(cnn_model.predict(x_test), axis=1)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Confusion matrix
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES, ax=axes[0])
axes[0].set_title('CNN Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Training curves
axes[1].plot(history_cnn.history['accuracy'], label='Train Accuracy')
axes[1].plot(history_cnn.history['val_accuracy'], label='Val Accuracy')
axes[1].plot(history_cnn.history['loss'], label='Train Loss', linestyle='--')
axes[1].plot(history_cnn.history['val_loss'], label='Val Loss', linestyle='--')
axes[1].set_title('CNN Training History')
axes[1].legend()
axes[1].set_xlabel('Epoch')

plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred_cnn, target_names=CIFAR10_CLASSES))
```

Repeat for the ResNet model with `resnet_model` and `history_resnet`.

This requires adding `scikit-learn` and `seaborn` to the pip install cell:

```bash
pip install scikit-learn seaborn
```

### 3.3 Add Model Saving and Loading (Section 1.8)

Replace the commented-out Google Drive save cell (cell 9) with:

```python
import os

MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

CNN_PATH = os.path.join(MODEL_DIR, 'cnn_cifar_model.keras')
RESNET_PATH = os.path.join(MODEL_DIR, 'resnet_cifar_model.keras')

# Save after training
cnn_model.save(CNN_PATH)
resnet_model.save(RESNET_PATH)
print(f"Models saved to {MODEL_DIR}/")
```

Then modify the training cells (Steps 3 and 4) to skip training if a saved model exists:

```python
if os.path.exists(CNN_PATH):
    print("Loading pre-trained CNN model...")
    cnn_model = keras.models.load_model(CNN_PATH)
    test_loss, test_acc = cnn_model.evaluate(x_test, y_test, verbose=0)
    print(f"CNN Test Accuracy: {test_acc*100:.2f}%")
else:
    print("Building and training CNN model...")
    cnn_model = build_cnn_model()
    # ... training code ...
```

### 3.4 Implement Soft Voting Ensemble (Section 2.2)

Modify the Kafka message schema to include the full softmax probability vector, not just the top prediction:

```python
# In both CNN and ResNet agents, change the response to include all probabilities:
response = {
    'agent': 'CNN-CIFAR',
    'image_id': image_id,
    'true_label': request['true_label'],
    'true_class': request['true_class'],
    'predicted_class': result['predicted_class'],
    'predicted_name': result['predicted_name'],
    'confidence': result['confidence'],
    'probabilities': predictions.tolist(),  # <-- ADD THIS: full 10-class vector
    'inference_time_ms': inference_time * 1000,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}
```

Then update the Judge's comparison logic:

```python
# Replace the simple confidence comparison with soft voting:
for img_id, results in comparison_map.items():
    if 'cnn' in results and 'resnet' in results:
        images_with_both_responses += 1

        cnn_probs = np.array(results['cnn']['probabilities'])
        resnet_probs = np.array(results['resnet']['probabilities'])
        true_label = results['cnn']['true_label']

        # Soft voting: average the probability vectors
        ensemble_probs = (cnn_probs + resnet_probs) / 2.0
        ensemble_pred = np.argmax(ensemble_probs)

        if ensemble_pred == true_label:
            brokered_total_correct += 1
```

### 3.5 Add a Transfer Learning Agent (Section 2.1)

Add a new cell after Step 4 to build a MobileNetV2-based model:

```python
def build_mobilenet_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None  # ImageNet weights expect 96x96 minimum; train from scratch on 32x32
    )

    model = models.Sequential([
        layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
        layers.RandomRotation(0.05),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

For true transfer learning with pre-trained weights, resize CIFAR-10 images to 96x96:

```python
x_train_resized = tf.image.resize(x_train, (96, 96))
x_test_resized = tf.image.resize(x_test, (96, 96))

def build_mobilenet_transfer():
    base_model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze pre-trained layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

Add corresponding Kafka topic and agent (follow the same pattern as Steps 5-6):

```python
MOBILENET_RESPONSE_TOPIC = 'mobilenet_classifications'
```

Update the Judge to consume from three response topics instead of two.

### 3.6 Add a Live Dashboard with Streamlit (Section 2.3)

Create a separate file `dashboard.py`:

```python
import streamlit as st
import json
import time
from kafka import KafkaConsumer
from collections import defaultdict
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Agent Performance Dashboard", layout="wide")
st.title("Image Classification Agent Dashboard")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []

# Create consumers for all response topics
@st.cache_resource
def get_consumers():
    topics = ['cnn_classifications', 'resnet_classifications']
    consumers = {}
    for topic in topics:
        consumers[topic] = KafkaConsumer(
            topic,
            bootstrap_servers=['localhost:9092'],
            group_id=f'dashboard_{topic}_{int(time.time())}',
            auto_offset_reset='latest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=1000
        )
    return consumers

# Poll for new messages
consumers = get_consumers()
for topic, consumer in consumers.items():
    for message in consumer:
        st.session_state.results.append(message.value)

# Build dataframe and display metrics
if st.session_state.results:
    df = pd.DataFrame(st.session_state.results)
    df['correct'] = df['predicted_class'] == df['true_label']

    col1, col2 = st.columns(2)
    for agent, col in zip(df['agent'].unique(), [col1, col2]):
        agent_df = df[df['agent'] == agent]
        accuracy = agent_df['correct'].mean() * 100
        col.metric(f"{agent} Accuracy", f"{accuracy:.1f}%", f"n={len(agent_df)}")

    # Per-class accuracy heatmap
    st.subheader("Per-Class Accuracy")
    pivot = df.pivot_table(index='true_class', columns='agent', values='correct', aggfunc='mean')
    fig = px.imshow(pivot, text_auto='.0%', color_continuous_scale='Greens')
    st.plotly_chart(fig, use_container_width=True)
```

Run with: `streamlit run dashboard.py`

This requires: `pip install streamlit plotly`

### 3.7 Containerize with Docker Compose (Section 2.6)

Create a `docker-compose.yml` at the project root:

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  jupyter:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8888:8888"
    depends_on:
      - kafka
    volumes:
      - ./:/workspace
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
```

Create a `Dockerfile`:

```dockerfile
FROM tensorflow/tensorflow:2.19.0-gpu-jupyter

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

Create `requirements.txt`:

```
kafka-python
matplotlib
pillow
tqdm
scikit-learn
seaborn
```

Update the notebook to read the Kafka bootstrap server from an environment variable:

```python
import os
KAFKA_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
```

Then replace all `bootstrap_servers=['localhost:9092']` references with `bootstrap_servers=KAFKA_SERVERS`.

Run with:

```bash
docker compose up -d
```

### 3.8 Export Results and Add Statistical Tests (Section 2.10)

Add a cell after the visualization step:

```python
import csv
from scipy import stats

# Export all results to CSV
all_results = []
for img_id, results in comparison_map.items():
    if 'cnn' in results and 'resnet' in results:
        row = {
            'image_id': img_id,
            'true_label': results['cnn']['true_label'],
            'true_class': results['cnn']['true_class'],
            'cnn_prediction': results['cnn']['predicted_name'],
            'cnn_confidence': results['cnn']['confidence'],
            'cnn_correct': int(results['cnn']['predicted_class'] == results['cnn']['true_label']),
            'cnn_time_ms': results['cnn']['inference_time_ms'],
            'resnet_prediction': results['resnet']['predicted_name'],
            'resnet_confidence': results['resnet']['confidence'],
            'resnet_correct': int(results['resnet']['predicted_class'] == results['resnet']['true_label']),
            'resnet_time_ms': results['resnet']['inference_time_ms'],
        }
        all_results.append(row)

df = pd.DataFrame(all_results)
df.to_csv('classification_results.csv', index=False)
print(f"Exported {len(df)} results to classification_results.csv")

# McNemar's test: are the two models significantly different?
# Build contingency table: both right, CNN right/ResNet wrong, etc.
both_right = ((df['cnn_correct'] == 1) & (df['resnet_correct'] == 1)).sum()
cnn_only = ((df['cnn_correct'] == 1) & (df['resnet_correct'] == 0)).sum()
resnet_only = ((df['cnn_correct'] == 0) & (df['resnet_correct'] == 1)).sum()
both_wrong = ((df['cnn_correct'] == 0) & (df['resnet_correct'] == 0)).sum()

contingency = [[both_right, cnn_only], [resnet_only, both_wrong]]
result = stats.contingency.chi2_contingency([contingency[0], contingency[1]], correction=True)
print(f"\nMcNemar's Test:")
print(f"  CNN correct, ResNet wrong: {cnn_only}")
print(f"  ResNet correct, CNN wrong: {resnet_only}")
print(f"  Chi-squared: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")
if result.pvalue < 0.05:
    print("  Result: Models are SIGNIFICANTLY different (p < 0.05)")
else:
    print("  Result: No significant difference between models (p >= 0.05)")
```

This requires: `pip install scipy pandas`
