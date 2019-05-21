from neuralstyle.stylize import stylize
import cv2
# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

content_image = cv2.imread('videos/mehdi.png')
style_images = [cv2.imread('neuralstyle/examples/1-style.jpg'), ]
print(style_images)
output_path = 'out1.png'
initial = content_image
initial_noiseblend = None
preserve_colors = None
iterations = ITERATIONS
content_weight = CONTENT_WEIGHT
content_weight_blend = CONTENT_WEIGHT_BLEND
style_weight = STYLE_WEIGHT
style_layer_weight_exp = STYLE_LAYER_WEIGHT_EXP
style_blend_weights = [1,]
tv_weight = TV_WEIGHT
learning_rate = LEARNING_RATE
beta1 = BETA1
beta2 = BETA2
epsilon = EPSILON
pooling = POOLING
print_iterations= None
checkpoint_iterations = None

for iteration, image, loss_vals in stylize(
        network=VGG_PATH,
        initial=initial,
        initial_noiseblend=initial_noiseblend,
        content=content_image,
        styles=style_images,
        preserve_colors=preserve_colors,
        iterations=iterations,
        content_weight=content_weight,
        content_weight_blend=content_weight_blend,
        style_weight=style_weight,
        style_layer_weight_exp=style_layer_weight_exp,
        style_blend_weights=style_blend_weights,
        tv_weight=tv_weight,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        pooling=pooling,
        print_iterations=print_iterations,
        checkpoint_iterations=checkpoint_iterations,
    ):
	print(loss_vals)