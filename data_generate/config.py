batch_size=1
use_lpips_loss=100
resnet_image_size=256
model_res=1024

img_path='../dataset/face_area'
# generated_images_dir='../dataset/generate/front'
# dlatent_dir='../dataset/dlatent/front'

result_dir='../dataset/result'

feature_direction_path='../data_regression/data/feature_direction_10_12.pkl'


model_choose='resnet'

cache_dir='./model'
vgg_model_url='./model/vgg16.pkl'
load_resnet='./model/finetuned_resnet.h5'
load_effnet='./model/finetuned_effnet.h5'
model_url='./model/stylegan_model.pkl'

