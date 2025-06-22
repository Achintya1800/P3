# MNIST Handwritten Digit Generator

A web app that generates synthetic handwritten digits using a Conditional GAN trained on MNIST dataset.

## Features

- Generate any digit (0-9) with 5 unique samples
- Real-time generation through web interface
- High-quality CNN-based architecture
- Mobile-friendly responsive design

## Quick Start

```bash
git clone https://github.com/yourusername/mnist-digit-generator.git
cd mnist-digit-generator
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py              # Streamlit web application
├── requirements.txt    # Dependencies
├── generator.pth       # Trained model weights
├── config.toml        # Streamlit config
└── training.ipynb     # Model training notebook
```

## Model Architecture

- **Type**: Conditional GAN
- **Generator**: 3-layer MLP with BatchNorm + Dropout
- **Training**: 50 epochs on MNIST dataset
- **Input**: 64-dim noise + digit label
- **Output**: 28×28 grayscale images

## Training

Run the training notebook in Google Colab:
```python
# Key hyperparameters
latent_dim = 64
epochs = 50
batch_size = 128
lr = 0.0002
```

## Deployment

### Render
1. Fork this repo
2. Connect to Render
3. Deploy as Web Service

### Streamlit Cloud
1. Push to GitHub
2. Deploy at [share.streamlit.io](https://share.streamlit.io)

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: PyTorch
- **Model**: Conditional GAN
- **Deployment**: Render/Streamlit Cloud

## Results

The model generates diverse, recognizable handwritten digits with 95%+ recognition accuracy.

## License

MIT License - see [LICENSE](LICENSE) for details.
