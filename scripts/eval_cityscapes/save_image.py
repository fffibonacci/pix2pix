import os
import sys
import caffe
import argparse
import numpy as np
import scipy.misc
from PIL import Image
from util import *
from cityscapes import cityscapes
labels = __import__('labels')

parser = argparse.ArgumentParser()
parser.add_argument("--cityscapes_dir", type=str, required=True, help="Path to the original cityscapes dataset")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--output_dir", type=str, required=True, help="Where to save the evaluation results")
parser.add_argument("--caffemodel_dir", type=str, default='./scripts/eval_cityscapes/caffemodel/', help="Where the FCN-8s caffemodel stored")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu id to use")
parser.add_argument("--split", type=str, default='val', help="Data split to be evaluated")
parser.add_argument("--save_output_images", type=int, default=0, help="Whether to save the FCN output images")
args = parser.parse_args()

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_output_images > 0:
        output_image_dir = args.output_dir + 'image_outputs/'
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
    CS = cityscapes(args.cityscapes_dir)
    n_cl = len(CS.classes)
    label_frames = CS.list_label_frames(args.split)
#    caffe.set_device(args.gpu_id)
#    caffe.set_mode_gpu()
#    net = caffe.Net(args.caffemodel_dir + 'deploy.prototxt',
#                    args.caffemodel_dir + 'fcn-8s-cityscapes.caffemodel',
#                    caffe.TEST)

#    hist_perframe = np.zeros((n_cl, n_cl))
    for i, idx in enumerate(label_frames):
       # if( i>10 ):
       #     break
        if i % 10 == 0:
            print('Saving: %d/%d' % (i, len(label_frames)))
        city = idx.split('_')[0]
        # idx is city_shot_frame
        label = CS.load_label(args.split, city, idx)
        im_file = args.result_dir + '/' + idx + '_fake_B.png'
        im = np.array(Image.open(im_file))
#        im = im[:,:,0:3]
#        print(im.dtype, label.dtype)
#        im = scipy.misc.imresize(im, (256, 256),interp='nearest')
        #im_label = np.zeros((256,256))
        # change prediction image from color to label using neighbor 
        #for i in range(256):
        #    for j in range(256):
        #        color = im[i][j]
        #        im_label[i][j] = neighbor_id(color)
        #im_label = im_label[np.newaxis, ...]       
        #out = segrun(net, CS.preprocess(im_label))
        # print(im_label.flatten().shape) 
        #im_label = scipy.misc.imresize(im_label,(1024,2048))
        #np.int8!!!!!overflow
	#label = np.squeeze(label)
        #im_label = np.array(im_label,dtype = np.uint8)
	#index = np.where((im_label.flatten()==13) & (label.flatten()==13))
	#print(index)
        #false_im = (im_label==label)*255
        #hist_perframe += fast_hist(label.flatten(), im_label.flatten(), n_cl)
	im = scipy.misc.imresize(im,(1024,2048))
        if args.save_output_images > 0:
            #label_im = CS.palette(label)
            #pred_im = CS.palette(im_label)
            scipy.misc.imsave(output_image_dir + '/' + idx + '_resize_color.jpg',im)
            #scipy.misc.imsave(output_image_dir + '/' + idx + '_gt.jpg', label_im)
            #scipy.misc.imsave(output_image_dir + '/' + idx + '_input_trainId.jpg', im_label)
            #scipy.misc.imsave(output_image_dir + '/' + idx + '_3gt.jpg', im)
            #scipy.misc.imsave(output_image_dir + '/' + idx + '_false.jpg', false_im)
    #mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
   # with open(args.output_dir + '/evaluation_results.txt', 'w') as f:
   #     f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
   #     f.write('Mean class accuracy: %f\n' % mean_class_acc)
   #     f.write('Mean class IoU: %f\n' % mean_class_iou)
   #     f.write('************ Per class numbers below ************\n')
   #     for i, cl in enumerate(CS.classes):
   #         while len(cl) < 15:
   #             cl = cl + ' '
   #         f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))
  
def neighbor_id(color):
    min_dist = 1000
    min_id = -1
    for i in range(len(labels.labels)):
        a = labels.labels[i].color - color
        dist = abs(a[0]) + abs(a[1]) + abs(a[2])
        #min_dist = min(min_dist, dist)
        if(min_dist > dist):
             min_dist = dist
             min_id = labels.labels[i].trainId
  	
    return min_id

main()
