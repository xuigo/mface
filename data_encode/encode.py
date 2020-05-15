import os
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config as config
from encoder.generator_model_resnet import Generator
from encoder.perceptual_model_resnet import PerceptualModel,load_images
from keras.models import load_model
#import feature_axis.feature_axis as feature_axis
import glob
import time
#from face_alignment.face_align import face_align

tflib.init_tf()
with dnnlib.util.open_url(config.model_url, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
generator = Generator(Gs_network, config.batch_size, clipping_threshold=2.0, 
tiled_dlatent=False, model_res=config.model_res, randomize_noise=False)

perc_model = None
if (config.use_lpips_loss > 0.00000001):
    with dnnlib.util.open_url(config.vgg_model_url, cache_dir=config.cache_dir) as f:
        perc_model =  pickle.load(f) 
 #* CNN model init *#
ff_model = None
if config.model_choose=='resnet':           
    ff_model = load_model(config.load_resnet)
    from keras.applications.resnet50 import preprocess_input
    preprocess_input=preprocess_input
elif config.model_choose=='efficientnet':
    import efficientnet
    print("Loading EfficientNet Model:")
    ff_model = load_model(config.load_effnet)
    from efficientnet import preprocess_input
    preprocess_input=preprocess_input
else:
    raise KeyError('model choose is erro .')
perceptual_model = PerceptualModel(perc_model=perc_model, batch_size=config.batch_size)
perceptual_model.build_perceptual_model(generator)

def latent_ext(images_batch,names):       
    perceptual_model.set_reference_images(images_batch)
    dlatents = None
    if (ff_model is not None): # predict initial dlatents with ** model
        dlatents = ff_model.predict(preprocess_input(load_images(images_batch,image_size=config.resnet_image_size)))
    if dlatents is not None:
        generator.set_dlatents(dlatents)
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=100)
    pbar = tqdm(op, leave=False, total=100)
    vid_count = 0
    best_loss = None
    best_dlatent = None
    for loss_dict in pbar:
        pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v)
                for k, v in loss_dict.items()]))
        if best_loss is None or loss_dict["loss"] < best_loss:
            best_loss = loss_dict["loss"]
            best_dlatent = generator.get_dlatents()            
        generator.stochastic_clip_dlatents()
    print(" ".join(names), " Loss {:.4f}".format(best_loss))
    # Generate images from found dlatents and save them
    generator.set_dlatents(best_dlatent)
    generated_images = generator.generate_images()
    generated_dlatents = generator.get_dlatents()
    os.makedirs(config.generated_images_dir,exist_ok=True)
    os.makedirs(config.dlatent_dir,exist_ok=True)
    for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(config.generated_images_dir, f'{img_name}_effi.png'), 'PNG')
        np.save(os.path.join(config.dlatent_dir, f'{img_name}.npy'), dlatent)         
        #self.latent_guide(img_name,dlatent)
    generator.reset_dlatents()

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__=='__main__':
    ref_images = [os.path.join(config.img_path, x) for x in os.listdir(config.img_path)]
    ref_images = list(filter(os.path.isfile, ref_images))
    if len(ref_images) == 0:
        raise Exception('%s is empty' % self.config.img_path)
    for images_batch in tqdm(split_to_batches(ref_images, config.batch_size), 
        total=len(ref_images)//config.batch_size):        
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
            latent_ext(images_batch,names)
   
