import argparse
import sys

import tensorflow as tf

from models.dense_net import DenseNet
from data_providers.utils import get_data_provider_by_name

FLAGS = None

train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 10, ## original: 300
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}


def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn

def main(_):
    
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        #### oss: here using only one gpu, need to change to use two
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d/gpu:0" % \
                                                  FLAGS.task_index,cluster=cluster)):

            model_params = vars(FLAGS)
            train_params = get_train_params_by_name(FLAGS.dataset)
            print("Prepare training data...")
            data_provider = get_data_provider_by_name(FLAGS.dataset, train_params)
            print("Initialize the model..")
            model = DenseNet(data_provider=data_provider, **model_params)
            global_step = tf.contrib.framework.get_or_create_global_step()

            train_op = model.train_step

        # The StopAtStepHook handles stopping after running given steps.
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        n_epochs = train_params['n_epochs']

        hooks=[tf.train.StopAtStepHook(last_step=n_epochs)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        step_counter=0
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(FLAGS.task_index == 0),checkpoint_dir="/tmp/train_logs",hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                if step_counter == reduce_lr_epoch_1 or step_counter == reduce_lr_epoch_2:
                    model.learning_rate = model.learning_rate / 10
                    print("Decrease learning rate, new lr = %f" % learning_rate)
                    # Run a training step asynchronously.
                    # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
                    # perform *synchronous* training.
                    # mon_sess.run handles AbortedError in case of preempted PS.
                    # if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                    #     train_params['initial_learning_rate'] = train_params['initial_learning_rate'] / 10
                mon_sess.run(train_op)
                if step_counter%(n_epochs//10)==0:
                    model.save_model(mon_sess)
                step_counter+=1

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument("--ps_hosts",\
                        type=str,\
                        default="",\
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts",\
                        type=str,\
                        default="",\
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--job_name",\
                        type=str,\
                        default="",\
                        help="One of 'ps', 'worker'")
    # Flags for defining the tf.train.Server
    parser.add_argument("--task_index",\
                        type=int,\
                        default=0,\
                        help="Index of task within the job")

    #### DenseNet flags
    parser.add_argument('--train', action='store_true',help='Train the model')
    parser.add_argument('--test', action='store_true',\
                        help='Test model for required dataset if pretrained model exists.'
                        'If provided together with `--train` flag testing will be'
                        'performed right after training.')
    parser.add_argument('--model_type', '-m',\
                        type=str,\
                        choices=['DenseNet', 'DenseNet-BC'],\
                        default='DenseNet',\
                        help='What type of model to use')
    parser.add_argument('--growth_rate',\
                        '-k', type=int,\
                        choices=[12, 24, 40],\
                        default=12,\
                        help='Grows rate for every layer, '\
                        'choices were restricted to used in paper')
    parser.add_argument('--depth', '-d', type=int,\
                        choices=[40, 100, 190, 250],default=40,\
                        help='Depth of whole network, restricted to paper choices')
    parser.add_argument('--dataset', '-ds', type=str,\
                        choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN'],\
                        default='C10',\
                        help='What dataset should be used')
    parser.add_argument('--total_blocks', '-tb', type=int,\
                        default=3, metavar='',\
                        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument('--keep_prob', '-kp', type=float,\
                        metavar='',help="Keep probability for dropout.")
    parser.add_argument('--weight_decay', '-wd', type=float,\
                        default=1e-4, metavar='',\
                        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument('--nesterov_momentum', '-nm',\
                        type=float, default=0.9, metavar='',\
                        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')

    parser.add_argument('--logs', dest='should_save_logs', action='store_true',help='Write tensorflow logs')
    parser.add_argument('--no-logs', dest='should_save_logs', action='store_false',help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument('--saves', dest='should_save_model', action='store_true',help='Save model during training')
    parser.add_argument('--no-saves', dest='should_save_model', action='store_false', help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    parser.add_argument('--renew-logs', dest='renew_logs', action='store_true',help='Erase previous logs for model if exists.')
    parser.add_argument('--not-renew-logs', dest='renew_logs', action='store_false',help='Do not erase previous logs for model if exists.')
    
    parser.add_argument('--num_inter_threads', '-inter', type=int, default=1, metavar='',help='number of inter threads for inference / test')
    parser.add_argument('--num_intra_threads', '-intra', type=int, default=128, metavar='',help='number of intra threads for inference / test')
    
    parser.set_defaults(renew_logs=True)

    FLAGS, unparsed = parser.parse_known_args()

    if not FLAGS.keep_prob:
        if FLAGS.dataset in ['C10', 'C100', 'SVHN']:
            FLAGS.keep_prob = 0.8
        else:
            FLAGS.keep_prob = 1.0
    if FLAGS.model_type == 'DenseNet':
        FLAGS.bc_mode = False
        FLAGS.reduction = 1.0
    elif FLAGS.model_type == 'DenseNet-BC':
        FLAGS.bc_mode = True

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    #### put it on ps
    with tf.device(tf.train.replica_device_setter(ps_tasks=len(ps_hosts))):
        
        model_params = vars(FLAGS)

        if not FLAGS.train and not FLAGS.test:
            print("You should train or test your network. Please check params.")
            exit()

            # some default params dataset/architecture related
        train_params = get_train_params_by_name(FLAGS.dataset)
        print("Params:")
        for k, v in model_params.items():
            print("\t%s: %s" % (k, v))
            print("Train params:")
        for k, v in train_params.items():
            print("\t%s: %s" % (k, v))
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
