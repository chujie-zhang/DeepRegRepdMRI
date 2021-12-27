
#47
import tensorflow as tf
import util.layers as layer
import util.helpers as helper
import util.losses as loss
import util.utils as util
class BaseNet:
    def __init__(self, minibatch_size, image_moving, image_fixed):
        self.minibatch_size = minibatch_size
        self.image_size = image_fixed.shape.as_list()[1:4]
        self.image_moving = image_moving
        self.image_fixed = image_fixed
        self.input_layer = tf.concat([layer.resize_volume(image_moving, self.image_size), image_fixed], axis=4)
        
        self.input_layer_moving = image_moving
        self.input_layer_fixed = image_fixed

        self.grid_ref = util.get_reference_grid(self.image_size)

class CnnNet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        self.transform_initial = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
        
        h0, hc0 = layer.downsample_resnet_block(self.input_layer, 2, 4, k_conv0=[7, 7, 7], k_pool = [1, 2, 2, 2, 1],name='global_down_0')
        h1, hc1 = layer.downsample_resnet_block(h0, 4, 8, k_pool = [1, 2, 2, 2, 1], name='global_down_1')
        h2, hc2_moving = layer.downsample_resnet_block(h1, 8, 16, k_pool = [1, 2, 2, 2, 1],name='global_down_2')
        h3, hc3 = layer.downsample_resnet_block(h2, 16, 16, k_pool = [1, 2, 2, 2, 1],name='global_down_3')
        h4, hc4 = layer.downsample_resnet_block(h3, 16, 16, k_pool = [1, 2, 2, 2, 1],name='global_down_4')
        h5, hc5 = layer.downsample_resnet_block(h4, 16, 16, k_pool = [1, 2, 2, 2, 1],name='global_down_5')
        h6, hc6 = layer.downsample_resnet_block(h5, 16, 8, k_pool = [1, 2, 2, 2, 1],name='global_down_6')
        h7, hc7 = layer.downsample_resnet_block(h6, 8, 4, k_pool = [1, 2, 2, 2, 1],name='global_down_7')
        h9 = layer.conv3_block(h7, 4, 2,  name='global_deep_8')

        
        self.fixed_feature = tf.gather(h9,indices=[1],axis=4)
        self.moving_feature =tf.gather(h9,indices=[0],axis=4)
        
        #self.theta_predicted, self.phi_predicted = loss.similarity(self.fixed_feature, self.moving_feature)
        
        #self.theta, self.phi = loss.similarity(self.input_layer_fixed, self.input_layer_moving)
        
        
        theta = layer.fully_connected(h9, 12, self.transform_initial, name='global_project_0')
        
        self.grid_warped = util.warp_grid(self.grid_ref, theta)
        self.ddf = self.grid_warped - self.grid_ref


'''
#43
import tensorflow as tf
import util.layers as layer
import util.helpers as helper
import util.losses as loss
import util.utils as util
class BaseNet:
    def __init__(self, minibatch_size, image_moving, image_fixed):
        self.minibatch_size = minibatch_size
        self.image_size = image_fixed.shape.as_list()[1:4]
        self.image_moving = image_moving
        self.image_fixed = image_fixed
        self.input_layer = tf.concat([layer.resize_volume(image_moving, self.image_size), image_fixed], axis=4)
        
        self.input_layer_moving = image_moving
        self.input_layer_fixed = image_fixed

        self.grid_ref = util.get_reference_grid(self.image_size)

class CnnNet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        self.transform_initial = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
        
        h0, hc0 = layer.downsample_resnet_block(self.input_layer, 2, 8, k_conv0=[7, 7, 7], k_pool = [1, 2, 2, 2, 1],name='global_down_0')
        h1, hc1 = layer.downsample_resnet_block(h0, 8, 8, k_pool = [1, 2, 2, 2, 1], name='global_down_1')
        h2, hc2_moving = layer.downsample_resnet_block(h1, 8, 8, k_pool = [1, 2, 2, 2, 1],name='global_down_2')
        h3, hc3 = layer.downsample_resnet_block(h2, 8, 4, k_pool = [1, 2, 2, 2, 1],name='global_down_3')
        h4, hc4 = layer.downsample_resnet_block(h3, 4, 4, k_pool = [1, 2, 2, 2, 1],name='global_down_4')
        h5, hc5 = layer.downsample_resnet_block(h4, 4, 4, k_pool = [1, 2, 2, 2, 1],name='global_down_5')
        h6, hc6 = layer.downsample_resnet_block(h5, 4, 4, k_pool = [1, 2, 2, 2, 1],name='global_down_6')
        h7, hc7 = layer.downsample_resnet_block(h6, 4, 4, k_pool = [1, 2, 2, 2, 1],name='global_down_7')
        h9 = layer.conv3_block(h7, 4, 2,  name='global_deep_8')

        
        self.fixed_feature = tf.gather(h9,indices=[1],axis=4)
        self.moving_feature =tf.gather(h9,indices=[0],axis=4)
        
        #self.theta_predicted, self.phi_predicted = loss.similarity(self.fixed_feature, self.moving_feature)
        
        #self.theta, self.phi = loss.similarity(self.input_layer_fixed, self.input_layer_moving)
        
        
        theta = layer.fully_connected(h9, 12, self.transform_initial, name='global_project_0')
        
        self.grid_warped = util.warp_grid(self.grid_ref, theta)
        self.ddf = self.grid_warped - self.grid_ref
'''
