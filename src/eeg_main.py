#!/usr/bin/env python3

from absl import logging, flags, app
import collections
import sh
import time
import random
import os
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import sklearn
import IPython

import dataset
import models

from torchcontrib.optim import SWA
from two_sample_distance import pdist


flags.DEFINE_integer('batch_size', 32, '')
#flags.DEFINE_integer('hop', 10, 'eeg samples stride for train set')
flags.DEFINE_float('hop_in_ms', 15, 'eeg stride for train set [ms]')

flags.DEFINE_string('optim', 'Adam', '')
flags.DEFINE_float('learning_rate',3e-4, '')
flags.DEFINE_float('laplace_smoothing', 1e-2, 'for class weights, as a fraction of num_classes')
flags.DEFINE_float('teacher_forcing_ratio', 1., '')

flags.DEFINE_integer('gpus', 1, '')
flags.DEFINE_integer('epochs', 20, '')
flags.DEFINE_integer('num_mel_centroids', 12, '')

flags.DEFINE_bool('debug', False, '')
flags.DEFINE_bool('clean_logs_dir', False, '')
flags.DEFINE_bool('final_eval', False, '')
flags.DEFINE_bool('SWA', True, '')
flags.DEFINE_integer('swa_start', 100, '')


flags.DEFINE_bool('OLS', False, '')
flags.DEFINE_bool('DenseModel', False, '')

flags.DEFINE_bool('discretize_MFCCs', False, '')



flags.DEFINE_bool('mixed_loss', False, '')


flags.DEFINE_multi_integer('lr_milestones',[90,110],'epochs where lr is decreased.')


FLAGS = flags.FLAGS


def main(_):
    if FLAGS.clean_logs_dir:
        sh.rm('-r', '-f', 'logs')
        sh.mkdir('logs')
    if not torch.cuda.is_available():
        FLAGS.gpus = 0
        torch.Tensor.cuda = lambda x: x

    if FLAGS.gpus:
        time.sleep(5)

    if not FLAGS.patient_eight:
        FLAGS.num_mel_centroids=10


    if FLAGS.OLS or FLAGS.DenseModel: #make sure output length is 1.
        assert FLAGS.use_MFCCs==True, "OLS can so far only be used with MFCCs"
        FLAGS.window_size=50
        print("Running with OLS/Dense. Re-setting window_size to 50ms")


    train_ds = dataset.get_data(split='train', hop=FLAGS.hop_in_ms)

    test_ds = dataset.get_data(split='test')
    logging.info(f'train size: {len(train_ds)}, test size: {len(test_ds)}')
    num_classes = train_ds.num_audio_classes
    sampling_rate_audio = round(FLAGS.sampling_rate_eeg * train_ds.audio_eeg_sample_ratio)

    if not FLAGS.use_MFCCs:
        class_freqs = torch.histc(train_ds.audio.float(), bins=num_classes, min=0, max=num_classes-1)
        class_freqs += FLAGS.laplace_smoothing * num_classes
        class_weights = 1. / class_freqs
        class_weights /= class_weights.sum()
        val_acc_for_mode = (train_ds.audio == test_ds.audio.mode()[0].item()).float().mean()
        logging.info(f'Validation accuracy when predicting mode: {val_acc_for_mode}')

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.input_shape = train_ds[0][0].shape # first batch of EEG data #seq_len_input, num_channels

            self.mel_transformer=train_ds.tacotron_mel_transformer
            
            if FLAGS.discretize_MFCCs:
                global_k_means_quantization=True # If True, centroids from kmeans across all bins are used. If False, centroids from within bins are used.
                if global_k_means_quantization:
                    self.mel_spec_discretizer=dataset.GlobalMelSpecDiscretizer()
                else:
                    self.mel_spec_discretizer=dataset.LocalMelSpecDiscretizer()

            if FLAGS.use_MFCCs:
                self.output_shape=self.mel_transformer.mel_spectrogram(train_ds[0][1].unsqueeze(0)).squeeze(0).T.shape #seq_len x mel bins ??
                if FLAGS.discretize_MFCCs:
                    class_weights=None
                    #class_weights=torch.load('class_weights_median.pt')
                    if class_weights is None:
                        self.criterion=torch.nn.CrossEntropyLoss(reduction='none') #combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
                    else:
                        self.criterion=torch.nn.CrossEntropyLoss(weight=class_weights,reduction='none') 
                else:
                    self.criterion = torch.nn.MSELoss(reduction='none')

            else:
                self.output_shape = train_ds[0][1].shape + (num_classes,) #first batch of audio data
                self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')

            if FLAGS.OLS:
                self.seq2seq = models.OLSModel(self.input_shape, self.output_shape)
            elif FLAGS.DenseModel:
                self.seq2seq = models.DensenetModel()
            else:
                self.seq2seq = models.RNNSeq2Seq(self.input_shape, self.output_shape)

        def loss(self, logits, y):
            if not FLAGS.use_MFCCs:
                y = y.flatten() # [bs x seq_len] -> [bs*seq_len] @YK: WHY??
                logits = logits.flatten(0, 1) #[bs x seq_len x num_classes] -> [bs*seq_len x num_classes]
            if FLAGS.discretize_MFCCs:
                current_batch_size=logits.shape[0] #will be flags.batchsize almost always but not in the very end of validation/train
                logits=logits.reshape((current_batch_size,-1,80,FLAGS.num_mel_centroids)) # bs x seq_len x 80 x 12
                logits=logits.transpose(1,3).transpose(2,3)
            return self.criterion(logits, y) #[bs * seq_len]

        def logits_to_classes(self,logits):
            current_batch_size=logits.shape[0] #will be flags.batchsize almost always but not in the very end of validation/train
            logits=logits.reshape((current_batch_size,-1,80,FLAGS.num_mel_centroids)) # bs x seq_len x 80 x 12
            logits=torch.argmax(logits,dim=3) #bs x seq_len x 80. each entry is the class.
            return logits

        def logits_to_mel_centroids(self,logits):
            logits=self.logits_to_classes(logits)
            return self.mel_spec_discretizer.class_to_centroids(logits)

        def contrastive_loss(self, x, encoder_outputs):
            #encoder_outputs [bs x seq_len_after_conv x hidden_size * directions] 
            #x [bs x seq_len_before_conv x channels]

            patient_4=(encoder_outputs[x[:,0,214]==0,:,:])
            patient_8=(encoder_outputs[x[:,0,214]==1,:,:])

            if len(patient_4<FLAGS.batch_size) and len(patient_8<FLAGS.batch_size):
                distance_within_patient_4=torch.mean(torch.nn.functional.pdist(patient_4.flatten(1,2))) #the flatten is not nice but very fast
                distance_within_patient_8=torch.mean(torch.nn.functional.pdist(patient_8.flatten(1,2)))
                distance_across_patients=torch.mean(pdist(patient_4.flatten(1,2),patient_8.flatten(1,2)))  #(n_1, d),(n_2,d)
                delta=1
                return (distance_within_patient_4+distance_within_patient_8+distance_across_patients)**2
            else:
                return 0

        def accuracy(self, logits, y, topk=1):
            if FLAGS.use_MFCCs: # y and logits are [bs x num_frames x num_bins]
                if FLAGS.discretize_MFCCs:
                    #1. actual accrutacy
                    class_predictions=self.logits_to_classes(logits)
                    accuracy=1.0*torch.sum(y==class_predictions)/y.numel()

                    #2. soft (differentiable) pearson r:
                    if FLAGS.mixed_loss:
                        current_batch_size=logits.shape[0] #will be flags.batchsize almost always but not in the very end of validation/train
                        logits_in_softmax=logits.reshape((current_batch_size,-1,80,FLAGS.num_mel_centroids)) # bs x seq_len x 80 x 12
                        logits_in_softmax=torch.nn.functional.softmax(logits_in_softmax,dim=-1) # bs x seq_len x 80 x 12
                        y_in_softmax=torch.nn.functional.one_hot(y).float()

                        logits_in_softmax=logits_in_softmax.flatten(2,3)
                        y_in_softmax=y_in_softmax.flatten(2,3)
                        soft_pearson_r=torch.nn.functional.cosine_similarity(y_in_softmax-torch.mean(y_in_softmax,dim=1).unsqueeze(1),logits_in_softmax-torch.mean(logits_in_softmax,dim=1).unsqueeze(1),dim=1)   
                    else:
                        soft_pearson_r=accuracy.new_zeros((1))

                    #3. pearson r:
                    logits=self.logits_to_mel_centroids(logits)
                    y=self.mel_spec_discretizer.class_to_centroids(y)
                    pearson_r=torch.nn.functional.cosine_similarity(y-torch.mean(y,dim=1).unsqueeze(1),logits-torch.mean(logits,dim=1).unsqueeze(1),dim=1)   


                    return pearson_r,accuracy, soft_pearson_r #mean is taken later
                else:
                    pearson_r=torch.nn.functional.cosine_similarity(y-torch.mean(y,dim=1).unsqueeze(1),logits-torch.mean(logits,dim=1).unsqueeze(1),dim=1)   
                    return pearson_r
            else:
                _, topi = torch.topk(logits, k=topk, dim=-1)
                return (y.unsqueeze(-1) == topi).float().sum(-1)

        def forward(self, x):
            pass

        def training_step(self, batch, batch_idx):
            if self.trainer.current_epoch>=FLAGS.epochs-2 and FLAGS.SWA:
                for param_group in self.trainer.optimizers[0].param_groups:
                            param_group['lr'] = 0.000000000000000001 #precent sgd from walking away from averaged point

            x, y = batch
            if FLAGS.use_MFCCs:
                y=self.mel_transformer.mel_spectrogram(y).transpose(1,2)
                if FLAGS.discretize_MFCCs:
                    y=self.mel_spec_discretizer.mel_to_class(y)
            teacher_forcing = torch.bernoulli(x.new_ones((x.shape[0],)) * FLAGS.teacher_forcing_ratio).byte()
           
            logits,attn_matrix, encoder_outputs = self.seq2seq(x, y=y, teacher_forcing=teacher_forcing)
            loss = self.loss(logits, y)

            if FLAGS.double_trouble:
                contrastive_loss=self.contrastive_loss(x,encoder_outputs)
                loss= loss + 0.1*contrastive_loss

            if FLAGS.discretize_MFCCs:
                pearson_r,acc, soft_pearson_r= self.accuracy(logits, y) #first is pearson_r
            else:
                acc = self.accuracy(logits, y)

            if not FLAGS.use_MFCCs:
                acc5 = self.accuracy(logits, y, 5)

            if FLAGS.use_MFCCs: 
                for param_group in self.trainer.optimizers[0].param_groups:
                    current_lr=(param_group['lr'])
                if FLAGS.discretize_MFCCs:
                    logs = {'loss/train': loss.mean(), 'pearson_r/train': pearson_r.mean(), 'accuracy/train': acc.mean(), 'learning_rate': current_lr}
                else:
                    logs = {'loss/train': loss.mean(), 'pearson_r/train': acc.mean(), 'learning_rate': current_lr}
                if batch_idx == 0:
                    if FLAGS.discretize_MFCCs:
                        logits=self.logits_to_mel_centroids(logits)
                        y=self.mel_spec_discretizer.class_to_centroids(y)
                    MFCC_plot = dataset.create_MFCC_plot(logits[0],y[0]) #passt first sample
                    self.logger.experiment.add_image('MFCC_plot/train', MFCC_plot, global_step=self.global_step, dataformats='HWC')
                    attn_plot = models.create_attention_plot(attn_matrix)
                    self.logger.experiment.add_image('Attention_Matrix', attn_plot, global_step=self.global_step, dataformats='HWC')


            else:
                logs = {'loss/train': loss.mean(), 'acc/acc': acc.mean(), 'acc5': acc5.mean()}
                if batch_idx == 0:
                    audio_idx = random.randint(0, len(batch) - 1)
                    audio_real = dataset.audio_classes_to_signal_th(y[audio_idx])
                    audio_pred = dataset.audio_classes_to_signal_th(logits[audio_idx].argmax(-1))
                    self.logger.experiment.add_audio(
                            tag='audio_real',
                            snd_tensor=audio_real.unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=sampling_rate_audio,
                            )
                    self.logger.experiment.add_audio(
                            tag='audio_pred',
                            snd_tensor=audio_pred.unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=sampling_rate_audio,
                            )

                    audio_plot = dataset.create_audio_plot([
                            (audio_real, 'real'),
                            (audio_pred, 'predicted_tf_{}'.format(teacher_forcing[audio_idx].item())),
                            ])
                    self.logger.experiment.add_image('audio_plot', audio_plot, global_step=self.global_step, dataformats='HWC')
            #return {**logs, 'log': logs}
            if FLAGS.mixed_loss:
                my_lambda=0.75
                return {'loss': my_lambda* logs['loss/train'] - (1-my_lambda)*soft_pearson_r.mean(), 'log': logs}
            else:
                return {'loss': logs['loss/train'], 'log': logs}



        def validation_step(self, batch, batch_idx):
            x, y = batch
            if FLAGS.use_MFCCs:
                y=self.mel_transformer.mel_spectrogram(y).transpose(1,2)
                if FLAGS.discretize_MFCCs:
                    y=self.mel_spec_discretizer.mel_to_class(y) # bs x seq_len (num mel frames) x mel_bins
            logits,attn_matrix,encoder_outputs = self.seq2seq(x)  # no teacher forcing
            loss = self.loss(logits, y)
            if FLAGS.discretize_MFCCs:
                pearson_r,acc,soft_pearson_r = self.accuracy(logits, y) # in OLS/DenseNet the MFCC frames are not a sequence. 
            else:
                acc= self.accuracy(logits,y)
            acc5=acc.new_zeros((1,1))

            if FLAGS.use_MFCCs:
                every_kth=1#int(1024/256)
                if FLAGS.discretize_MFCCs:
                    y=self.mel_spec_discretizer.class_to_centroids(y)
                    logits=self.logits_to_mel_centroids(logits)
                    outs = {'loss': loss, 'acc': pearson_r, 'acc5': acc5, 'targets': y[:,::every_kth,:],'predictions': logits[:,::every_kth,:],'actual_acc': acc.unsqueeze(0)} #unsq. for consistency in later cat.
                else:
                    outs = {'loss': loss, 'acc': acc, 'acc5': acc5, 'targets': y[:,::every_kth,:],'predictions': logits[:,::every_kth,:]}
                #if batch_idx == 0:
                 #   outs['MFCC_plot_val'] = dataset.create_MFCC_plot(logits[0],y[0])
            else:
                outs = {'loss': loss, 'acc': acc, 'acc5': acc5}
                if batch_idx == 0:
                    audio_idx = random.randint(0, len(batch) - 1)
                    outs['val_audio_real'] = dataset.audio_classes_to_signal_th(y[audio_idx])
                    outs['val_audio_pred'] = dataset.audio_classes_to_signal_th(logits[audio_idx].argmax(-1))
                    outs['val_audio_plot'] = dataset.create_audio_plot([
                            (outs['val_audio_real'], 'real'),
                            (outs['val_audio_pred'], 'predicted'),
                            ])
            return outs

        def validation_epoch_end(self, outputs): #outputs contains outs of validation_step for entire val set.
            outs = collections.defaultdict(list)
            for o in outputs:
                for k, v in o.items(): 
                    outs[k].append(v)


            outputs = {k: torch.cat(v, 0) for k, v in outs.items()}
            #acc = outputs['acc'].mean()

            all_predictions=outputs['predictions'].view((outputs['predictions'].shape[0]*outputs['predictions'].shape[1],outputs['predictions'].shape[2]))
            all_targets=outputs['targets'].view((outputs['targets'].shape[0]*outputs['targets'].shape[1],outputs['targets'].shape[2]))            
            if FLAGS.discretize_MFCCs: #acc function cannot handle this input if discretization is used.
                acc=torch.nn.functional.cosine_similarity(all_predictions-torch.mean(all_predictions,dim=1).unsqueeze(1),all_targets-torch.mean(all_targets,dim=1).unsqueeze(1),dim=1)   
                acc=acc.mean()
            else:
                acc= self.accuracy(all_predictions,all_targets).mean()

            acc5 = outputs['acc5'].mean()
            if FLAGS.discretize_MFCCs:
                actual_acc = outputs['actual_acc'].mean()

                #torch.save(all_predictions.T,f"mel_preds/EPOCH50bs_{FLAGS.batch_size}_lr_{FLAGS.learning_rate}_tfr_{FLAGS.teacher_forcing_ratio}_ws_{FLAGS.window_size}_emb_dm_{FLAGS.hidden_size}.wav.pt")
            if self.trainer.current_epoch==FLAGS.epochs-2:
                torch.save(all_predictions.T,f"mel_preds/bs_{FLAGS.batch_size}_lr_{FLAGS.learning_rate}_tfr_{FLAGS.teacher_forcing_ratio}_ws_{FLAGS.window_size}_emb_dm_{FLAGS.hidden_size}_do_{FLAGS.dropout}_pnpnd_{FLAGS.pre_and_postnet_dim}.wav.pt")
                torch.save(all_targets.T,f"mel_preds/TARGETS_bs_{FLAGS.batch_size}_lr_{FLAGS.learning_rate}_tfr_{FLAGS.teacher_forcing_ratio}_ws_{FLAGS.window_size}_emb_dm_{FLAGS.hidden_size}_do_{FLAGS.dropout}_pnpnd_{FLAGS.pre_and_postnet_dim}.wav.pt")

                if FLAGS.SWA:
                    self.trainer.optimizers[0].swap_swa_sgd()
                    print("SWITCHING TO SWA WEIGHTS")
            #IPython.embed()
            #torch.save(all_targets.T,'mel_preds/new_targets'+ str(self.trainer.current_epoch))
            if self.trainer.current_epoch==FLAGS.epochs-1 and FLAGS.SWA:
                torch.save(all_predictions.T,f"mel_preds/AFTER_SWA_bs_{FLAGS.batch_size}_lr_{FLAGS.learning_rate}_tfr_{FLAGS.teacher_forcing_ratio}_ws_{FLAGS.window_size}_emb_dm_{FLAGS.hidden_size}.wav.pt")


            if FLAGS.use_MFCCs:
                self.logger.experiment.add_image('MFCC_plot/val', dataset.create_MFCC_plot(all_predictions,all_targets), global_step=self.global_step, dataformats='HWC')
            else:
                for tag in ('val_audio_real', 'val_audio_pred'):
                    self.logger.experiment.add_audio(
                            tag=tag,
                            snd_tensor=outputs[tag].unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=sampling_rate_audio,
                            )
                self.logger.experiment.add_image('val_audio_plot', outputs['val_audio_plot'], global_step=self.global_step, dataformats='HWC')

            if FLAGS.use_MFCCs: 
                if FLAGS.discretize_MFCCs:
                    logs = {
                            'loss/val': outputs['loss'].mean(),
                            'pearson_r/val': acc,
                            'accuracy/val':actual_acc
                            }
                else:
                    logs = {
                        'loss/val': outputs['loss'].mean(),
                        'pearson_r/val': acc,
                        }
            else:
                logs = {
                        'loss/val': outputs['loss'].mean(),
                        'acc/val_acc': acc,
                        'val_acc_rel_vs_rand': acc / (1. / num_classes),
                        'val_acc_rel_vs_mode': acc / val_acc_for_mode,
                        'val_acc5': acc5,
                        'val_acc5_rel_vs_rand': acc5 / (1. / num_classes),
                        'val_acc5_rel_vs_mode': acc5 / val_acc_for_mode,
                        }
            return {'val_loss': logs['loss/val'], 'log': logs}

        def test_step(self, batch, batch_idx, dataloader_idx):

            x, y = batch
            if FLAGS.use_MFCCs:
                y=self.mel_transformer.mel_spectrogram(y).transpose(1,2)
                if FLAGS.discretize_MFCCs:
                    y=self.mel_spec_discretizer.mel_to_class(y)

            logits, attn_matrix, encoder_outputs = self.seq2seq(x)  # no teacher forcing
            preds = logits.argmax(-1)
            return {
                    'audio_real': dataset.audio_classes_to_signal_th(y).flatten(), 
                    'audio_pred': dataset.audio_classes_to_signal_th(preds).flatten()}

        def test_epoch_end(self, dataloader_outputs):
            for outputs, name in zip(dataloader_outputs, ('train', 'test')):
                audio_real = torch.cat([o['audio_real'] for o in outputs], 0).cpu().numpy()
                audio_pred = torch.cat([o['audio_pred'] for o in outputs], 0).cpu().numpy()
                # TODO: save audio

                os.makedirs("results",exist_ok=True)
                np.save(f'results/{name}_bs_{FLAGS.batch_size}',np.concatenate((audio_real,audio_pred)))

            return {} # has to be here

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                    train_ds, 
                    shuffle=True,
                    drop_last=True,
                    num_workers=3,
                    batch_size=FLAGS.batch_size)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(
                    test_ds, 
                    shuffle=False,
                    drop_last=False,
                    num_workers=3,
                    batch_size=FLAGS.batch_size)

        def test_dataloader(self):
            train_ds_full_hop = dataset.get_data('train')
            return [torch.utils.data.DataLoader(ds, shuffle=False, drop_last=False, num_workers=3,
                    batch_size=FLAGS.batch_size) for ds in (train_ds_full_hop, test_ds)]

        def configure_optimizers(self): 
            optim = next(o for o in dir(torch.optim) if o.lower() == FLAGS.optim.lower()) # "Adam"
            optimizer=getattr(torch.optim, optim)(self.parameters(), lr=FLAGS.learning_rate) # optimizer object
            #optimizer=torch.optim.SGD(self.parameters(), lr=FLAGS.learning_rate,weight_decay=0.00001) 
            optimizer=torch.optim.AdamW(self.parameters(),lr=FLAGS.learning_rate,weight_decay=0.00001)
            if FLAGS.SWA:
                iterations_per_epoch=int(len(train_ds)/FLAGS.batch_size)
                optimizer = SWA(optimizer, swa_start=int(FLAGS.swa_start*iterations_per_epoch), swa_freq=50, swa_lr=FLAGS.learning_rate/10)
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=FLAGS.lr_milestones, gamma=0.5)
            #scheduler=torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=FLAGS.learning_rate/2,max_lr=2*FLAGS.learning_rate,step_size_up=2,step_size_down=2,cycle_momentum=False)
            #step_size_up is in epochs! I don't know why the hell
            return [optimizer], [scheduler]

    
    model = Model()

    #run_on_gpu(cpu_allowed=True)


    trainer = pl.Trainer(
            gpus=FLAGS.gpus,
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            default_save_path='logs',
            logger=pl.loggers.TensorBoardLogger('logs',name=f"p13/31/bs {FLAGS.batch_size}, lr {FLAGS.learning_rate}, tfr {FLAGS.teacher_forcing_ratio}, ws {FLAGS.window_size}, lyrs {FLAGS.n_layers}, emb dm {FLAGS.hidden_size}, drpt {FLAGS.dropout},pnpnd {FLAGS.pre_and_postnet_dim} "),
            terminate_on_nan=True,
            row_log_interval=100,
            nb_sanity_val_steps=8,
            gradient_clip_val=2
            )



    trainer.fit(model)

    if FLAGS.final_eval:
        trainer.test(model)


if __name__ == '__main__':
    app.run(main)
