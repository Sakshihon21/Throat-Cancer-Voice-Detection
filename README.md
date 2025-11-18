# Throat Cancer Detection using Voice

A non-invasive AI project for early detection of throat cancer from voice recordings.

## Features
- Extracts MFCC, Chroma, and Spectral features from voice samples
- Uses CNN + LSTM hybrid model for classification
- Can be used for demo telemedicine or early screening

## How to Run
1. Install dependencies:
2. Place demo audio files in `data/healthy` and `data/cancer`.
3. Run training:
4.Model saved as `throat_cancer_model.h5