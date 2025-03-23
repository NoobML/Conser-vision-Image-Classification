import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm
import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D,Dropout,Concatenate, Flatten
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split



train = pd.read_csv('train_features.csv')
test = pd.read_csv('test_features.csv')
Y = pd.read_csv('train_labels.csv')
test_ids = test['id']


num_images = 5
sample_images = train.iloc[:num_images]

fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

for i, ax in enumerate(axes):
    img_path = sample_images.iloc[i]['filepath']
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Image {i+1}")
plt.show()


frac = 1.0
y = Y.sample(frac=frac, random_state=1)
x = train.loc[y.index, ['id', 'filepath', 'site']]

# 80% train, 20% validation
X_train, X_val, Y_train, Y_val = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=42
)

train_merged = X_train.copy()
train_merged[Y_train.columns] = Y_train
val_merged = X_val.copy()
val_merged[Y_val.columns] = Y_val


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, image_size=(224, 224), df='train'):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size
        self.df = df

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.dataframe.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Load images
        images = np.array([
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(filepath, target_size=self.image_size)
            ) / 255.0
            for filepath in batch_df['filepath']
        ])


        if self.df == 'train':
            labels = np.array(batch_df[["antelope_duiker", "bird", "blank", "civet_genet",
                                        "hog", "leopard", "monkey_prosimian", "rodent"]], dtype=np.float32)
            return images, labels

        if self.df == 'test':
            return images, np.zeros((len(images), 8))

def build_model(image_shape, num_classes=8, lambda_reg=0.1):
    image_input = Input(shape=image_shape, name="image_input")

    X = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(lambda_reg))(image_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(lambda_reg))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(lambda_reg))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(lambda_reg))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(lambda_reg))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(lambda_reg))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = GlobalAveragePooling2D()(X)

    X = Dense(units=128, activation='relu', kernel_regularizer=l2(lambda_reg))(X)
    output = Dense(units=num_classes, activation='softmax',
                   kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(lambda_reg))(X)

    model = Model(inputs=[image_input], outputs=output)
    return model

learning_rate = 0.2  # Adjust as needed

model = build_model(image_shape=(224, 224, 3))
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryCrossentropy(name="log_loss")])

print(model.summary())

train_generator = CustomDataGenerator(train_merged, batch_size=32, df='train')
valid_generator = CustomDataGenerator(val_merged, batch_size=32, df='train')
test_generator =  CustomDataGenerator(test, batch_size=32, df='test')

model.load_weights("model_6_weights.keras")

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint("model_6_weights.keras", monitor="val_loss", save_best_only=True, mode="min")

with tf.device('/GPU:0'):
    history = model.fit(train_generator, epochs=100, validation_data=valid_generator, callbacks=[checkpoint, reduce_lr])

model.save_weights("model_6_weights.keras")

preds = model.predict(test_generator, verbose=1)
class_names = ["antelope_duiker", "bird", "blank", "civet_genet", "hog", "leopard", "monkey_prosimian", "rodent"]
df_preds = pd.DataFrame(preds, columns=class_names)
df_preds['id'] = test_ids
df_preds = df_preds[['id'] + [col for col in df_preds.columns if col != 'id']]
df_preds.to_csv('submission1.csv', index=False)

def plot_log_loss(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['loss'], 'bo-', label='Training Log Loss')
    plt.plot(epochs, history.history['val_loss'], 'r*-', label='Validation Log Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss (Binary Crossentropy)')
    plt.legend()
    plt.title('Training vs Validation Log Loss')
    plt.show()

plot_log_loss(history)
