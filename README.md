# Drop-out
Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

## Dataset
> you can access to the dataset from [here](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data)

## model 

```
 Layer (type)                Output Shape              Param #
=================================================================
 zero_padding2d (ZeroPadding  (None, 30, 30, 1)        0
 2D)

 conv2d (Conv2D)             (None, 28, 28, 32)        320

 dropout (Dropout)           (None, 28, 28, 32)        0

 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0
 )

 zero_padding2d_1 (ZeroPaddi  (None, 16, 16, 32)       0
 ng2D)

 conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496

 dropout_1 (Dropout)         (None, 14, 14, 64)        0

 max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0
 2D)

 flatten (Flatten)           (None, 3136)              0

 dense (Dense)               (None, 128)               401536

 dropout_2 (Dropout)         (None, 128)               0

 dense_1 (Dense)             (None, 10)                1290

=================================================================
Total params: 421,642
Trainable params: 421,642
Non-trainable params: 0
```
## Result 

```
Epoch 1/10
300/300 [==============================] - 97s 322ms/step - loss: 0.6746 - accuracy: 0.7606 - val_loss: 0.4264 - val_accuracy: 0.8578
Epoch 2/10
300/300 [==============================] - 96s 321ms/step - loss: 0.4172 - accuracy: 0.8523 - val_loss: 0.3785 - val_accuracy: 0.8663
Epoch 3/10
300/300 [==============================] - 105s 350ms/step - loss: 0.3669 - accuracy: 0.8707 - val_loss: 0.3232 - val_accuracy: 0.8854
Epoch 4/10
300/300 [==============================] - 89s 296ms/step - loss: 0.3357 - accuracy: 0.8806 - val_loss: 0.3123 - val_accuracy: 0.8884
Epoch 5/10
300/300 [==============================] - 87s 290ms/step - loss: 0.3130 - accuracy: 0.8880 - val_loss: 0.2962 - val_accuracy: 0.8977
Epoch 6/10
300/300 [==============================] - 87s 290ms/step - loss: 0.2956 - accuracy: 0.8941 - val_loss: 0.2711 - val_accuracy: 0.9006
Epoch 7/10
300/300 [==============================] - 86s 286ms/step - loss: 0.2834 - accuracy: 0.8986 - val_loss: 0.2619 - val_accuracy: 0.9064
Epoch 8/10
300/300 [==============================] - 86s 287ms/step - loss: 0.2687 - accuracy: 0.9033 - val_loss: 0.2563 - val_accuracy: 0.9059
Epoch 9/10
300/300 [==============================] - 87s 289ms/step - loss: 0.2578 - accuracy: 0.9080 - val_loss: 0.2531 - val_accuracy: 0.9102
Epoch 10/10
300/300 [==============================] - 86s 288ms/step - loss: 0.2458 - accuracy: 0.9114 - val_loss: 0.2589 - val_accuracy: 0.9064
```
