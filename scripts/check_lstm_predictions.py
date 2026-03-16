"""Cargar dataset y modelo LSTM y mostrar predicciones/metricas básicas."""
from __future__ import annotations
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'models')
DATA_PATH = os.path.join(MODEL_DIR, 'event_dataset.npz')
MODEL_H5 = os.path.join(MODEL_DIR, 'event_lstm.h5')
MODEL_DIR_KERAS = os.path.join(MODEL_DIR, 'lstm.keras')




def main():
    if not os.path.exists(DATA_PATH):
        print('Dataset not found:', DATA_PATH)
        return
    model_path = None
    if os.path.exists(MODEL_H5):
        model_path = MODEL_H5
    elif os.path.exists(MODEL_DIR_KERAS):
        model_path = MODEL_DIR_KERAS
    else:
        # try to find any keras model file or dir starting with 'event' or 'lstm'
        for fn in os.listdir(MODEL_DIR):
            p = os.path.join(MODEL_DIR, fn)
            if fn.endswith('.h5') or fn.endswith('.keras') or os.path.isdir(p) and fn.lower().startswith('lstm'):
                model_path = p
                break
    if model_path is None:
        print('No model artifact found in', MODEL_DIR)
        return

    data = np.load(DATA_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']
    ids = data['ids']
    print('Loaded dataset:', X.shape, 'labels:', y.shape, 'ids:', ids.shape)
    unique, counts = np.unique(y, return_counts=True)
    print('Label distribution:', dict(zip(unique.tolist(), counts.tolist())))

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except Exception as e:
        print('TensorFlow not available in this environment:', e)
        return

    model = load_model(model_path)
    print('Loaded model:', model_path)
    try:
        print('Model input shape:', model.input_shape)
    except Exception:
        pass

    # adapt features if model expects different channel dim
    expected_feat = None
    try:
        expected_feat = model.input_shape[-1]
    except Exception:
        expected_feat = None
    if expected_feat is not None and X.shape[-1] != expected_feat:
        print(f'Warning: dataset has {X.shape[-1]} features, model expects {expected_feat}. Attempting to broadcast.')
        if X.shape[-1] == 1 and expected_feat > 1:
            X = np.repeat(X, expected_feat, axis=2)
            print('Broadcasted X to', X.shape)
        else:
            print('Cannot automatically adapt feature dimension. Aborting predict.')
            return

    preds = model.predict(X, batch_size=128)
    yhat = np.argmax(preds, axis=1)
    acc = accuracy_score(y, yhat)
    print('Accuracy (on full dataset):', acc)
    print('\nClassification report:\n')
    print(classification_report(y, yhat, digits=4))

    # show sample predictions
    for i in range(min(10, len(y))):
        print(f"id={ids[i]} true={y[i]} pred={yhat[i]} probs={preds[i].tolist()}")


if __name__ == '__main__':
    main()
