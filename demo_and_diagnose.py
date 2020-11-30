import os
from torch.autograd import Variable
import math
import torch
import torch.utils.serialization

import matplotlib.pyplot as plt
import cv2

import random
import numpy as np
import numpy
import networks
from my_args import  args
from AverageMeter import  *
from skimage.measure import compare_ssim,compare_psnr
from skimage.color import rgb2yuv, yuv2rgb
from yuv_frame_io import YUV_Read,YUV_Write
torch.backends.cudnn.benchmark = False # True # to speed up the

HD720p_Other_DATA = "/content/drive/MyDrive/frame-interpolation/HD_dataset/HD720p_GT"
HD720p_Other_RESULT = "/content/drive/My Drive/frame-interpolation/hd-results/720p"

motion_dir = "./motion/"
kernel_dir = "./kernels/"
occlusion_dir = "./occlusion/"

if not os.path.exists(HD720p_Other_RESULT):
    os.mkdir(HD720p_Other_RESULT)

for path in [motion_dir, kernel_dir, occlusion_dir]:
    if not os.path.exists(path):
        os.mkdir(path)



model = networks.__dict__[args.netName](
                                channel=args.channels,
                                filter_size = args.filter_size ,
                                training=False)

if args.use_cuda:
    model = model.cuda()
args.SAVED_MODEL = './model_weights/' + args.SAVED_MODEL
print("The testing model weight is: " + args.SAVED_MODEL)
if not args.use_cuda:
    #pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
    model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
else:
    #pretrained_dict = torch.load(args.SAVED_MODEL)
    model.load_state_dict(torch.load(args.SAVED_MODEL))

model = model.eval() # deploy mode



def test_HD720p(model = model, use_cuda = args.use_cuda,save_which = args.save_which, dtype = args.dtype):
    files = sorted(os.listdir(HD720p_Other_DATA))
    unique_id =str(random.randint(0, 100000))
    gen_dir = os.path.join(HD720p_Other_RESULT, unique_id)
    os.mkdir(gen_dir)
    

    for file_i in files:
        print("\n\n\n**************")
        print(file_i)
        gen_file = os.path.join(HD720p_Other_RESULT, unique_id, file_i)
        input_file = os.path.join(HD720p_Other_DATA, file_i)

        interp_error = AverageMeter()
        psnr_error = AverageMeter()
        ssim_error = AverageMeter()

        print(input_file)
        print(gen_file)
        Reader = YUV_Read(input_file, 720, 1280, toRGB=True)
        Writer = YUV_Write(gen_file, fromRGB=True)
        for index in range(0, 100, 2): # len(files) - 2, 2):
            IMAGE1, sucess1 = Reader.read(index)
            IMAGE2, sucess2 = Reader.read(index + 2)
            
            if not sucess1 or not sucess2:
                print("Could not read frame")
                break

            X0 =  torch.from_numpy( np.transpose(IMAGE1 , (2,0,1)).astype("float32")/ 255.0).type(dtype)
            X1 =  torch.from_numpy( np.transpose(IMAGE2, (2,0,1)).astype("float32")/ 255.0).type(dtype)

            y_ = torch.FloatTensor()

            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))

            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if not channel == 3:
                continue


            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

            X0 = Variable(torch.unsqueeze(X0,0),volatile=True)
            X1 = Variable(torch.unsqueeze(X1,0), volatile=True)
            X0 = pader(X0)
            X1 = pader(X1)

            if use_cuda:
                X0 = X0.cuda()
                X1 = X1.cuda()
            y_s ,offset,filter,occlusion = model(torch.stack((X0, X1),dim = 0))
            y_ = y_s[save_which]

            if use_cuda:
                X0 = X0.data.cpu().numpy()
                y_ = y_.data.cpu().numpy()
                offset = [offset_i.data.cpu().numpy() for offset_i in offset]
                filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
                occlusion = [occlusion_i.data.cpu().numpy() for occlusion_i in occlusion] if occlusion[0] is not None else None
                X1 = X1.data.cpu().numpy()

            else:
                X0 = X0.data.numpy()
                y_ = y_.data.numpy()
                offset = [offset_i.data.numpy() for offset_i in offset]
                filter = [filter_i.data.numpy() for filter_i in filter]
                occlusion = [occlusion_i.data.numpy() for occlusion_i in occlusion]
                X1 = X1.data.numpy()



            X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
            y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
            offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
            filter = [np.transpose(
                filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for filter_i in filter]  if filter is not None else None
            occlusion = [np.transpose(
                occlusion_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for occlusion_i in occlusion]  if occlusion is not None else None
            X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

            print("y_", np.shape(y_))
            print("offset", np.shape(offset))
            print("filter", np.shape(filter))
            print("occlusion", np.shape(occlusion))


            # plot optical flow
            np_offset = np.asarray(offset)

            save_path = motion_dir + file_i.rsplit(".", 1)[0] + "/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            mag, angle = cv2.cartToPolar(np_offset[0,...,0], np_offset[0,...,1])
            mask = np.zeros((np_offset.shape[1], np_offset.shape[2], 3))
            # Sets image saturation to maximum 
            mask[..., 1] = 255
            # Sets image hue according to the optical flow  
            # direction 
            mask[..., 0] = angle * 180 / np.pi / 2
            # Sets image value according to the optical flow 
            # magnitude (normalized) 
            mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) 
            mask = mask.astype('uint8')
            # Converts HSV to RGB (BGR) color representation 
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR) 
            plt.figure()
            plt.title("Motion-" + str(index) + "-forward")
            plt.imshow(rgb)
            plt.savefig(save_path + str(index) + "_" + "forw.png")

            mag, angle = cv2.cartToPolar(np_offset[1,...,0], np_offset[1,...,1] )
            mask = np.zeros((np_offset.shape[1], np_offset.shape[2], 3))
            # Sets image saturation to maximum 
            mask[..., 1] = 255
            # Sets image hue according to the optical flow  
            # direction 
            mask[..., 0] = angle * 180 / np.pi / 2
            # Sets image value according to the optical flow 
            # magnitude (normalized) 
            mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) 
            mask = mask.astype('uint8')
            # Converts HSV to RGB (BGR) color representation 
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR) 
            plt.figure()
            plt.title("Motion-"+ str(index) + "-backward")
            plt.imshow(rgb)
            plt.savefig(save_path + str(index) + "_" + "back.png")


            # plot occlusion masks
            np_occlusion = np.asarray(occlusion)
            save_path = occlusion_dir + file_i.rsplit(".", 1)[0] + "/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.figure()
            plt.title("Occlusion-"+ str(index) + "-forward")
            plt.imshow(np_occlusion[0,:,:,0], cmap='gray')
            plt.savefig(save_path + str(index) + "_" + "forw.png")
            plt.figure()
            plt.title("Occlusion-"+ str(index) + "-backward")
            plt.imshow(np_occlusion[1,:,:,0], cmap='gray')
            plt.savefig(save_path + str(index) + "_" + "back.png")


            # plt.figure()
            # plt.title("Filters")
            # plt.imshow(filter.permute(1,2,0))
            # plt.savefig(kernel_dir + file_i + index +".png")
            # plt.figure()
            # plt.title("Occlusion")
            # plt.imshow(occlusion.permute(1,2,0))
            # plt.savefig(occlusion_dir + file_i + index +".png")

            Writer.write(IMAGE1)
            rec_rgb = np.round(y_).astype(numpy.uint8)
            Writer.write(rec_rgb)
            gt_rgb, sucess = Reader.read(index+1)
            gt_yuv = rgb2yuv(gt_rgb / 255.0)
            rec_yuv = rgb2yuv(rec_rgb / 255.0)

            gt_rgb = gt_yuv[:, :, 0] * 255.0
            rec_rgb = rec_yuv[:, :, 0] * 255.0

            gt_rgb = gt_rgb.astype('uint8')
            rec_rgb = rec_rgb.astype('uint8')

            diff_rgb = 128.0 + rec_rgb - gt_rgb
            avg_interp_error_abs = np.mean(np.abs(diff_rgb - 128.0))
            
            interp_error.update(avg_interp_error_abs,1)
            
            mse = numpy.mean((diff_rgb - 128.0) ** 2)
            if mse == 0:
                return 100.0
            PIXEL_MAX = 255.0
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            psnr_error.update(psnr, 1)
            
            psnr_ = compare_psnr(rec_rgb, gt_rgb)
            print(str(psnr) + '\t'+ str(psnr_))
            
            ssim = compare_ssim(rec_rgb, gt_rgb,multichannel=False)
            ssim_error.update(ssim,1)

            diff_rgb = diff_rgb.astype("uint8")

            print("interpolation error / PSNR : " + str(round(avg_interp_error_abs,4)) + " ,\t  psnr " + str(round(psnr,4))+ ",\t ssim " + str(round(ssim,5)))
            fh = open(os.path.join(HD720p_Other_RESULT, unique_id, file_i+ "_psnr_Y.txt"), "a+")
            fh.write(str(psnr))
            fh.write("\n")
            fh.close()
            fh = open(os.path.join(HD720p_Other_RESULT, unique_id, file_i+ "_ssim_Y.txt"), "a+")
            fh.write(str(ssim))
            fh.write("\n")
            fh.close()
            metrics = "The average interpolation error / PSNR for all images are : " + str(
                round(interp_error.avg, 4)) + ",\t  psnr " + str(round(psnr_error.avg, 4)) + ",\t  ssim " + str(
                round(ssim_error.avg, 4))
            print(metrics)


        metrics = "The average interpolation error / PSNR for all images are : " + str(round(interp_error.avg,4)) + ",\t  psnr " + str(round(psnr_error.avg,4)) + ",\t  ssim " + str(round(ssim_error.avg,4))
        print(metrics)
        fh = open(os.path.join(HD720p_Other_RESULT, unique_id, file_i+ "_psnr_Y.txt"), "a+")
        fh.write("\n")
        fh.write(str(psnr_error.avg))
        fh.write("\n")
        fh.close()
        fh = open(os.path.join(HD720p_Other_RESULT, unique_id, file_i+"_ssim_Y.txt"), "a+")
        fh.write("\n")
        fh.write(str(ssim_error.avg))
        fh.write("\n")
        fh.close()

            
if __name__ == '__main__':
    test_HD720p(model,args.use_cuda)
