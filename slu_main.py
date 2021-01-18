
from src.utils import init_experiment
from src.slu.dataloader import get_dataloader
from src.slu.trainer import SLUTrainer
from src.slu.model import BinarySLUTagger, SlotNamePredictor, SentRepreGenerator
from config import get_params

import numpy as np
from tqdm import tqdm

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

DEFAULT = 0
TEM_REG = 1
SLOT_REG = 2

def main(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    
    # get dataloader
    dataloader_tr, dataloader_val, dataloader_test, vocab = get_dataloader(params.tgt_dm, params.batch_size, params.tr, params.sr, params.n_samples)
    # set default device
    os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda_device

    # build model
    binary_slutagger = BinarySLUTagger(params, vocab)
    slotname_predictor = SlotNamePredictor(params)
    binary_slutagger, slotname_predictor = binary_slutagger.cuda(), slotname_predictor.cuda()
    if params.tr or params.sr:
        sent_repre_generator = SentRepreGenerator(params, vocab)
        sent_repre_generator = sent_repre_generator.cuda()
        slu_trainer = SLUTrainer(params, binary_slutagger, slotname_predictor, sent_repre_generator=sent_repre_generator)
        
    else:
        slu_trainer = SLUTrainer(params, binary_slutagger, slotname_predictor)
    
    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e+1))
        loss_bin_list, loss_slotname_list = [], []
        if params.tr:
            loss_tem0_list, loss_tem1_list = [], []
        elif params.sr:
            loss_right_st_list, loss_wrong_st_list = [], []
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        if params.tr:
            for i, (X, lengths, y_bin, y_final, y_dm, templates, tem_lengths) in pbar:
                X, lengths, templates, tem_lengths = X.cuda(), lengths.cuda(), templates.cuda(), tem_lengths.cuda()
                loss_bin, loss_slotname, loss_tem0, loss_tem1 = slu_trainer.train_step(X, lengths, y_bin, y_final, y_dm, templates=templates, tem_lengths=tem_lengths, epoch=e)
                loss_bin_list.append(loss_bin)
                loss_slotname_list.append(loss_slotname)
                loss_tem0_list.append(loss_tem0)
                loss_tem1_list.append(loss_tem1)

                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS TEM0:{:.4f} LOSS TEM1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_tem0_list), np.mean(loss_tem1_list)))
        
        elif params.sr:
            for i, (X, lengths, y_bin, y_final, y_dm, slot_type, st_lengths) in pbar:
                X, lengths, slot_type, st_lengths = X.cuda(), lengths.cuda(), slot_type.cuda(), st_lengths.cuda()
                # st_lengths: the number of slot types in each speech
                loss_bin, loss_slotname, loss_tem0, loss_tem1 = slu_trainer.train_step(X, lengths, y_bin, y_final, y_dm, slot_type=slot_type, slot_type_lengths=st_lengths, epoch=e)
                loss_bin_list.append(loss_bin)
                loss_slotname_list.append(loss_slotname)
                loss_right_st_list.append(loss_tem0)
                loss_wrong_st_list.append(loss_tem1)
                
                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS TYPE0:{:.4f} LOSS TYPE1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_right_st_list), np.mean(loss_wrong_st_list)))
        
        else:
            for i, (X, lengths, y_bin, y_final, y_dm) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss_bin, loss_slotname = slu_trainer.train_step(X, lengths, y_bin, y_final, y_dm)
                loss_bin_list.append(loss_bin)
                loss_slotname_list.append(loss_slotname)
                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS SLOT:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list)))
            
        if params.tr:
            logger.info("Finish training epoch {}. LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS TEM0:{:.4f} LOSS TEM1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_tem0_list), np.mean(loss_tem1_list)))
        elif params.sr:
            logger.info("Finish training epoch {}. LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS SLOT TYPE0:{:.4f} LOSS SLOT TYPE1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_right_st_list), np.mean(loss_wrong_st_list)))
        else:
            logger.info("Finish training epoch {}. LOSS BIN:{:.4f} LOSS SLOT:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list)))

        logger.info("============== Evaluate Epoch {} ==============".format(e+1))
        bin_f1, final_f1, stop_training_flag = slu_trainer.evaluate(dataloader_val, istestset=False)
        logger.info("Eval on dev set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, final_f1))

        bin_f1, final_f1, stop_training_flag = slu_trainer.evaluate(dataloader_test, istestset=True)
        logger.info("Eval on test set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, final_f1))

        if stop_training_flag == True:
            break


if __name__ == "__main__":
    params = get_params()
    main(params)