""" Report manager utility """
from __future__ import print_function

import sys
import time
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil

import numpy as np

from others.logging import logger


def build_report_manager(opt):
    report_mgr = ReportMgr()
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, tensorboard_dir):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """

        if os.path.isdir(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
            os.mkdir(tensorboard_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)


    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs) 


    def report_step(self, 
                    step,
                    D_losses,
                    G_losses, 
                    true_positives_tot,
                    true_positives_margin_tot,
                    total_positives_tot,
                    total_positives_margin_tot,
                    tp_tot,
                    fp_tot,
                    tn_tot,
                    fn_tot,
                    contradiction_found,
                    G_prediction_for_G_extraction,
                    G_prediction_for_W_extraction,
                    G_prediction_for_no_possible_extraction,
                    real_result,
                    fake_result):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self.D_losses=dict(D_losses)
        self.step=step
        self.D_losses=D_losses
        self.G_losses=G_losses
        self.true_positives_tot=true_positives_tot
        self.true_positives_margin_tot=true_positives_margin_tot
        self.total_positives_tot=total_positives_tot
        self.total_positives_margin_tot=total_positives_margin_tot
        self.tp_tot=tp_tot
        self.fp_tot=fp_tot
        self.tn_tot=tn_tot
        self.fn_tot=fn_tot
        self.contradiction_found=contradiction_found
        self.G_prediction_for_G_extraction=G_prediction_for_G_extraction
        self.G_prediction_for_W_extraction=G_prediction_for_W_extraction
        self.G_prediction_for_no_possible_extraction=G_prediction_for_no_possible_extraction
        self.real_result=real_result
        self.fake_result=fake_result

        self.log('Validation results at step %d' % (self.step))
        for key in self.D_losses.keys():
            if isinstance(self.D_losses[key],np.ndarray) and self.D_losses[key].shape and self.D_losses[key].shape[0]>1:
                self.log(key+' [%s] ' %(','.join(map(str, self.D_losses[key]))))
            else:
                self.log(key+' %g ' %(self.D_losses[key]))
        for key in self.G_losses.keys():
            self.log(key+' %g ' %(self.G_losses[key]))

        self.log('Found sentences:                          [{}] '.format(','.join(map(str, self.true_positives_tot))))
        self.log('Found sentences with margin:              [{}] '.format(','.join(map(str, self.true_positives_margin_tot))))
        self.log('Total sentences:                          [{}] '.format(','.join(map(str, self.total_positives_tot))))

        self.log('TP:                                       [{}] '.format(','.join(map(str, self.tp_tot))))
        self.log('FP:                                       [{}] '.format(','.join(map(str, self.fp_tot))))
        self.log('TN:                                       [{}] '.format(','.join(map(str, self.tn_tot))))
        self.log('FN:                                       [{}] '.format(','.join(map(str, self.fn_tot))))

        self.log('Found contradictions:                     [{}] '.format(','.join(map(str, self.contradiction_found))))

        self.log('G_prediction_for_G_extraction:            [{}] '.format(','.join(map(str, self.G_prediction_for_G_extraction))))
        self.log('G_prediction_for_W_extraction:            [{}] '.format(','.join(map(str, self.G_prediction_for_W_extraction))))
        self.log('G_prediction_for_no_possible_extraction:  [{}] '.format(','.join(map(str, self.G_prediction_for_no_possible_extraction))))

        self.log('Real examples spotted:                    [{}] '.format(','.join(map(str, self.real_result))))
        self.log('Fake examples spotted:                    [{}] '.format(','.join(map(str, self.fake_result))))

        # self.log_to_tensorboard()


    def log_to_tensorboard(self):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.G_prediction_for_G_extraction[1]==0:
            self.G_prediction_for_G_extraction[1]+=1
        if self.G_prediction_for_W_extraction[1]==0:
            self.G_prediction_for_W_extraction[1]+=1
        if self.G_prediction_for_no_possible_extraction[1]==0:
            self.G_prediction_for_no_possible_extraction[1]+=1
        
        D_sent_all_loss=self.D_losses.pop('D_sent_all_loss')
        self.writer.add_scalars('Losses/Model', self.D_losses, self.step)
        self.writer.add_scalars('Losses/Generator', self.G_losses, self.step)

        self.writer.add_scalars('Sentences/First Part', {"Found":self.true_positives_tot[0],
                                                         "Found margin":self.true_positives_margin_tot[0],
                                                         "Total":self.total_positives_tot[0]
                                                         },self.step)
        self.writer.add_scalars('Sentences/Second Part', {"Found":self.true_positives_tot[1],
                                                          "Found margin":self.true_positives_margin_tot[1],
                                                          "Total":self.total_positives_tot[1]
                                                          },self.step)

        self.writer.add_scalars('Classification metrics/First Part', {"TP":self.tp_tot[0],
                                                                      "FP":self.fp_tot[0]
                                                                      },self.step)
        self.writer.add_scalars('Classification metrics/Second Part', {"TP":self.tp_tot[1],
                                                                       "FP":self.fp_tot[1]
                                                                       },self.step)

        self.writer.add_scalars('Contradictions/Contradictions', {"Found":self.contradiction_found[0],
                                                                  "Total":self.contradiction_found[1]
                                                                 },self.step)

        self.writer.add_scalars('Contradictions/Quality doc prediction', {"GG":self.G_prediction_for_G_extraction[0]/self.G_prediction_for_G_extraction[1]*100,
                                                                          "GW":self.G_prediction_for_W_extraction[0]/self.G_prediction_for_W_extraction[1]*100,
                                                                          "GN":self.G_prediction_for_no_possible_extraction[0]/self.G_prediction_for_no_possible_extraction[1]*100,
                                                                         },self.step)

        self.writer.flush()
        self.log('Logging to tensorboard...')


    def end_report(self):
        self.writer.flush()
        self.writer.close()