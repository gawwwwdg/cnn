"""


WORKS WITHOUT APPLYING ANY FEATURE DECTECTORS/FILTERS/CONVOLUTION LAYERS 


MAYBE THE CASE OF OVERFITTING



"""
import keras
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0

test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10,batch_size=32)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)



