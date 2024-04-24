import time
import scipy 
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from pdb import set_trace as st
from tqdm import tqdm

def train(opt):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        if opt.which_epoch == 'latest':
            try:
                start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
                print(start_epoch, epoch_iter)
            except:
                start_epoch, epoch_iter = 1, 0
        else:
            start_epoch, epoch_iter = int(opt.which_epoch), 0

        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
        for update_point in opt.decay_epochs:
            if start_epoch < update_point:
                break

            opt.lr *= opt.decay_gamma
    else:
        start_epoch, epoch_iter = 0, 0

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)

    total_steps = (start_epoch) * dataset_size + epoch_iter

    bSize = opt.batchSize

    for epoch in range(start_epoch, opt.epochs):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = 0
        for i, data in enumerate(tqdm(dataset, total=len(dataset)), start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_inputs(data)
            disc_losses = model.update_D()
            gen_losses, gen_in, gen_out, rec_out, cyc_out = model.update_G(infer=save_fake)
            loss_dict = dict(gen_losses, **disc_losses)

            errors = {k: v.item() if not (isinstance(v, float) or isinstance(v, int)) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch+1, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            ### display output images
            if save_fake and opt.display_id > 0:
                class_a_suffix = ' class {}'.format(data['A_class'][0])
                class_b_suffix = ' class {}'.format(data['B_class'][0])
                classes = None

                visuals = OrderedDict()
                visuals_A = OrderedDict([('real image' + class_a_suffix, util.tensor2im(gen_in.data[0]))])
                visuals_B = OrderedDict([('real image' + class_b_suffix, util.tensor2im(gen_in.data[bSize]))])

                A_out_vis = OrderedDict([('synthesized image' + class_b_suffix, util.tensor2im(gen_out.data[0]))])
                B_out_vis = OrderedDict([('synthesized image' + class_a_suffix, util.tensor2im(gen_out.data[bSize]))])
                if opt.lambda_rec > 0:
                    A_out_vis.update([('reconstructed image' + class_a_suffix, util.tensor2im(rec_out.data[0]))])
                    B_out_vis.update([('reconstructed image' + class_b_suffix, util.tensor2im(rec_out.data[bSize]))])
                if opt.lambda_cyc > 0:
                    A_out_vis.update([('cycled image' + class_a_suffix, util.tensor2im(cyc_out.data[0]))])
                    B_out_vis.update([('cycled image' + class_b_suffix, util.tensor2im(cyc_out.data[bSize]))])

                visuals_A.update(A_out_vis)
                visuals_B.update(B_out_vis)
                visuals.update(visuals_A)
                visuals.update(visuals_B)

                ncols = len(visuals_A)
                visualizer.display_current_results(visuals, epoch, classes, ncols)

            ### save latest model
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch+1, total_steps))
            model.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            if opt.display_id == 0:
                model.eval()
                visuals = model.inference(sample_data)
                visualizer.save_matrix_image(visuals, 'latest')
                model.train()

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch+1, opt.epochs, time.time() - epoch_start_time))

        ### save model for this epoch
        if (epoch+1) % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch+1, total_steps))
            model.save('latest')
            model.save(epoch+1)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
            if opt.display_id == 0:
                model.eval()
                visuals = model.inference(sample_data)
                visualizer.save_matrix_image(visuals, epoch+1)
                model.train()

        ### multiply learning rate by opt.decay_gamma after certain iterations
        if (epoch+1) in opt.decay_epochs:
            model.update_learning_rate()

if __name__ == "__main__":
    opt = TrainOptions().parse()
    train(opt)
