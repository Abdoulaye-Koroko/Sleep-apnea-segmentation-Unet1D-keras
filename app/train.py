import numpy as np
import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from keras_applications.resnet_1d import ResNet18
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import F1Score
from segmentation_models.models.unet_1d import Unet1D

from app.data_utils import build_data
from app.metrics import recall_score,precision_score,f1_score




def train(args):
    
    db_folder = os.path.abspath("physionet.org/files/ucddb/1.0.0")
    X,y = build_data(db_folder,period=576)
    
    
    idx = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(idx, random_state = 111,test_size = 0.2)

    X_val = X[val_idx]
    X_train = X[train_idx] 
    y_val = y[val_idx]
    y_train = y[train_idx] 

    print(f"X_train:{X_train.shape}, X_val:{y_val.shape}, y_train:{y_train.shape}, y_val:{y_val.shape}")
    
    print(20*"====")
    
    print(f"Building model.....")
    
    n_signals = 14
    model = Unet1D(
        backbone_name='resnet18_1d',
        input_shape=(None, n_signals),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights=None,
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
)
    if args.optim=="adam":
        optim=Adam
    elif args.optim=="sgd":
        optim=SGD
    
    model.compile(loss=binary_crossentropy, 
              optimizer=optim(learning_rate=args.lr), 
              metrics=["accuracy",recall_score,precision_score,f1_score])
    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    save_path = os.path.abspath(f'models/model_{args.optim}_{args.lr}.h5')
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, mode='auto')
    
    model.fit(
    x=X_train,
    y=y_train,
    batch_size=args.batch_size,
    epochs=args.num_epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping,checkpoint]
)
    return




if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Function arguments')
    
    parser.add_argument('--optim', type=str, default="adam",
                        help='optimizer name')
    
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')

    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    train(args)
    
