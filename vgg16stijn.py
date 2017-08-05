import json
import numpy as np
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.optimizers import Adam, SGD, RMSprop
# from keras.callbacks import CSVLogger

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def RGB_to_BGR(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr
        
class Vgg16Stijn():
    def __init__(self): # sets up neural network
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create()
        self.get_classes()
    
    def create(self): # build actual network
        """
            Builds neural network and loads pretrained vgg16 model weights
        """
        model = self.model = Sequential()
        model.add(Lambda(RGB_to_BGR, input_shape=(3,224,224), output_shape=(3,224,224))) #change rgb to bgr
        
        self.add_convlayers(2, 64)
        self.add_convlayers(2, 128)
        self.add_convlayers(3, 256)
        self.add_convlayers(3, 512)
        self.add_convlayers(3, 512)
        
        model.add(Flatten())
        self.add_fullconnect()
        self.add_fullconnect()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
    
    def add_convlayers(self, layers, filters):
        """
            Adds a specified number of ZeroPadding (line of zeros around image) and Covolution layers
            to the model, and a MaxPooling (outputs max value of group) layer at the very end.

            Args:
                layers (int):   The number of zero padded convolution layers
                                to be added to the model.
                filters (int):  The number of convolution filters to be 
                                created for each layer. I think these are the 
                                number of learnable features per layer
        """
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    def add_fullconnect(self):
        """
            Adds a fully connected layer of 4096 neurons (with 64x64 dimension) to the model with a
            Dropout of 0.5. Dropout is random 0.5 of the input nodes are set to 0 to prevent overfitting.

            Args:   None
            Returns:   None
        """
        model = self.model
        model.add(Dense(4096, activation='relu')) # fully connected layer with 64x64 dimension
        model.add(Dropout(0.5)) # random half of the input nodes are set to 0 to prevent overfitting
    
    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        
    def batch_iterator(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(directory=path, target_size=(224,224),
                    class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def change_number_outputnodes(self, num):
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(num,activation='softmax'))
        self.compile()
        
    def compile(self, lr=0.001):
        """
            Configures the model for training.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.compile(optimizer=Adam(lr=lr),
                loss='binary_crossentropy', metrics=['accuracy'])
    
    def correct_class_id(self, batches):
        """
            Adjusts final layer to correct number of output nodes and updates class labels.
            
            Args:
                batches: keras.preprocessing.image.ImageDataGenerator opbject with flow_from_directory()
        """
        self.change_number_outputnodes(batches.nb_class)
        classes = list(iter(batches.class_indices)) 
        # creates list of classes from dict item 'batches.class_indices'
        # batches.class_indices i.e. {'invasive': 1, 'non_invasive': 0}
        for key in batches.class_indices:
            classes[batches.class_indices[key]] = key
        # orders class keys by value in dict like ['non_invasive', 'invasive'] because key 'non_invasive' has value 0
        self.classes = classes
        
    def fit(self, batches, val_batches, nb_epoch=1, extra_info=''):
        """
            Fits the model on data yielded batch-by-batch by a Python generator. 
            Saves metrics to extra_info+'training.log'.
            See Keras documentation: https://keras.io/models/model/
        """
#         csv_logger = CSVLogger(extra_info+'training.log', append=True)
#         weight_save_callback = ModelCheckpoint('/path/to/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,save_best_only=False, mode='auto')
#         model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,callbacks=[weight_save_callback])
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample) #, callbacks=[csv_logger])
    
    def fit_n_save_all(self, batches, val_batches, nb_epoch, results_path, extra_info=''):
        """
            Uses fit and saves weights every epoch
            
            Args:
                batches, val_batches, nb_epoch
                extra_info: string attached to filename
        """
        latest_weights_filename = ''
        for epoch in range(nb_epoch):
            print('Epoch %d' %epoch)
            self.fit(batches, val_batches, nb_epoch=1)
            latest_weights_filename = 'ft%d%s.h5' %(epoch+1, extra_info)
            self.model.save_weights(results_path+latest_weights_filename)
        print("Completed fit operations and saved in " + str(latest_weights_filename))
    
    def test(self, path, batch_size=8):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch.

            Args:
                path (string):  Path to the target directory. It should contain one subdirectory 
                                per class.
                batch_size (int): The number of images to be considered in each batch.
            
            Returns:
                test_batches, numpy array(s) of predictions for the test_batches.
    
        """
        test_batches = self.batch_iterator(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)