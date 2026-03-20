# Memory efficient ECG signal compression using Deep Auto-Encoder

## Overview
This project focuses on compressing single-lead ECG signals using a fully connected deep autoencoder built with Keras. The model learns a compact 2-dimensional latent representation from 36-point ECG segments, achieving an 18x compression ratio while preserving clinically relevant signal characteristics through a residual skip connection.

## Features
- ECG signal compression with 18x compression ratio (36 dimensions to 2)
- Signal reconstruction via symmetric decoder with residual connection
- Latent space extraction for compressed storage or transmission
- Performance evaluation (MSE,MAE, PRD, CR, QS)
- Batch Normalization and Dropout regularization for robust generalization
- Custom combined loss training with Adam optimizer

## Model Architecture
The autoencoder uses a symmetric encoder-decoder structure with a skip connection from input to output.

Encoder: 36 -> 32 -> 26 -> 20 -> 16 -> 8 -> 4 -> 2 (latent)

Decoder: 2 -> 4 -> 8 -> 16 -> 20 -> 26 -> 32 -> 36 (+ residual)

- L2 regularization (1e-5) on encoder layers
- Batch Normalization (momentum 0.99) after encoder dense layers
- Dropout (rate 0.1) at selected encoder layers
- Linear activation at bottleneck and output; ReLU elsewhere

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
1. Clone the repository:
```
git clone <repository-link>
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run:
```
jupyter notebook ECG_signal_compression.ipynb

```

## Usage
```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load model
model = tf.keras.models.load_model('ecg_autoencoder_model.h5')

# Load and window ECG signal
df = pd.read_csv('sample_ecg.csv')
signal = df['MLII'].values
window_size = 36
segments = np.array([signal[i:i+window_size] for i in range(0, len(signal) - window_size, window_size)])

# Normalize
scaler = MinMaxScaler()
segments_scaled = scaler.fit_transform(segments)

# Reconstruct
reconstructed = model.predict(segments_scaled)

# Extract latent representation
encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer('latent').output)
latent_codes = encoder.predict(segments_scaled)
print(f"Compressed shape: {latent_codes.shape}")  # (n_segments, 2)
```

## Project Structure
```
.
├── ecg_autoencoder_model.h5                       # Trained Keras autoencoder model
├── sample_ecg.csv                                 # Sample ECG signal (MLII lead, ~5000 samples)
├── ECG_signal_compression.ipynb                   # Entry point
├── requirements.txt                               # Python dependencies
└── README.md
```

## Notes
- The model was trained with a custom combined loss function. When loading the model, pass the custom loss via `custom_objects` if it is not registered globally.
- The residual Add layer adds the original input to the decoder output, helping preserve low-frequency signal trends during reconstruction.

## License
MIT
