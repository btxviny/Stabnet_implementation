import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

class Resnet(tf.keras.Model):
    def __init__(self, channels , trainable_layers):
        super(Resnet,self).__init__()
        resnet = ResNet50(include_top=False, weights='imagenet')
        #Get the dictionary of config for vgg16
        resnet_config = resnet.get_config()

        # Change the input shape to new desired shape
        h, w, c = None, None, channels
        resnet_config["layers"][0]["config"]["batch_input_shape"] = (None, h, w, c)


        #Create new model with the updated configuration
        resnet_updated = tf.keras.Model.from_config(resnet_config)

        # Function that calculates average of weights along the channel axis and then
        #copies it over n number of times. n being the new channels that need to be concatenated with the original channels. 
        def avg_and_copy_wts(weights, num_channels_to_fill):  #num_channels_to_fill are the extra channels for which we need to fill weights
            average_weights = np.mean(weights, axis=-2).reshape(weights[:,:,-1:,:].shape)  #Find mean along the channel axis (second to last axis)
            wts_copied_to_mult_channels = np.tile(average_weights, (num_channels_to_fill, 1)) #Repeat (copy) the array multiple times
            return(wts_copied_to_mult_channels)

        #Get the configuration for the updated model and extract layer names. 
        #We will use these names to copy over weights from the original model. 
        resnet_updated_config = resnet_updated.get_config()
        resnet_updated_layer_names = [resnet_updated_config['layers'][x]['name'] for x in range(len(resnet_updated_config['layers']))]

        #Name of the first convolutional layer.
        #Remember that this is the only layer with new additional weights. All other layers
        #will have same weights as the original model. 
        first_conv_name = resnet_updated_layer_names[2]

        #Update weights for all layers. And for the first conv layer, copy the first
        #three layer weights and fill others with the average of all three. 
        for layer in resnet.layers:
            if layer.name in resnet_updated_layer_names:
            
                if layer.get_weights() != []:  #All convolutional layers and layers with weights (no input layer or any pool layers)
                    target_layer = resnet_updated.get_layer(layer.name)
                
                    if layer.name in first_conv_name:    #For the first convolutionl layer
                        weights = layer.get_weights()[0]
                        biases  = layer.get_weights()[1]
                    
                        weights_extra_channels = np.concatenate((weights,   #Keep the first 3 channel weights as-is and copy the weights for additional channels.
                                                                avg_and_copy_wts(weights, c - 3)),  # - 3 as we already have weights for the 3 existing channels in our model. 
                                                                axis=-2)
                                                                
                        target_layer.set_weights([weights_extra_channels, biases])  #Now set weights for the first conv. layer
                        target_layer.trainable = False   #You can make this trainable if you want. 
                    
                    else:
                        target_layer.set_weights(layer.get_weights())   #Set weights to all other layers. 
                        target_layer.trainable = False  #You can make this trainable if you want. 

        for layer in resnet_updated.layers[:-trainable_layers]:
            layer.trainable = False
        for layer in resnet_updated.layers[-trainable_layers:]:
            layer.trainable = True
        self.resnet = resnet_updated

    def call(self,input):
        return self.resnet(input)