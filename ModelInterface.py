import utils as ut
import argparse
import copy
import sys, os
import numpy as np
import tensorflow as tf
# import tensorflow_io as tfio
import datetime
from tensorflow.keras import regularizers
from DeepFM import DeepFM
from XDeepFM import XDeepFM
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

import socket
hostname = socket.gethostname()
# if hostname == 'ml-p40-ser006.us01':
#    import tensorflow_io as tfio
    
import warnings
warnings.filterwarnings('ignore')


class Learner:
    def __init__(self, conf_yml, conf=None):
        self.conf_yml = conf_yml
        self.model = None
        if conf is None:
            self.conf = ut.load_yaml_file(conf_yml)
        else:
            self.conf = conf

    def get_fea_size(self):
        x = ''
        fea_size_file = self.conf['common']['fea_size_file']
        with open(fea_size_file) as f:
            x = f.read()
        return int(x.strip())

    def get_fea_slot(self):
        x = {}
        feature_size = 0
        fea_slot_file = self.conf['common']['fea_slot_file']
        with open(fea_slot_file) as f:
            for line in f:
                lis = line.strip().split('\t')
                a = int(lis[1])
                b = int(lis[2])
                feature_size = a + b
                x[lis[0]] = (a, b)
        return x, feature_size

    def get_preemb_feaslot(self, fea_slot_dic=None):
        if fea_slot_dic is None:
            fea_slot_dic, _ = self.get_fea_slot()
        preemb_fea_conf = self.conf['common']['preemb_fea_conf']
        res = []
        with open(preemb_fea_conf) as f:
            for line in f:
                k = line.strip()
                if k not in fea_slot_dic:
                    continue
                x = fea_slot_dic[k][0]
                y = x + fea_slot_dic[k][1]
                res.append([x, y])
        return res

    def get_multihot_feaslot(self, fea_slot_dic=None):
        if fea_slot_dic is None:
            fea_slot_dic, _ = self.get_fea_slot()
        mutihot_fea_conf = self.conf['common']['multihot_fea_conf']
        res = []
        with open(mutihot_fea_conf) as f:
            for line in f:
                k = line.strip()
                if k not in fea_slot_dic:
                    continue
                x = fea_slot_dic[k][0]
                y = x + fea_slot_dic[k][1]
                res.append([x, y])
        return res

    def get_multi_feaslot(self, fea_slot_dic=None):
        if fea_slot_dic is None:
            fea_slot_dic, _ = self.get_fea_slot()
        mutihot_fea_conf = self.conf['common']['multi_fea_conf']
        res = []
        with open(mutihot_fea_conf) as f:
            for line in f:
                k = line.strip()
                if k not in fea_slot_dic:
                    continue
                x = fea_slot_dic[k][0]
                y = x + fea_slot_dic[k][1]
                res.append([x, y])
        return res

    def get_padding_size(self, fea_slot, multihot_feaslot, fix_addition=0):
        x = len(fea_slot)
        if fix_addition > 0:
            return x + fix_addition
        for y in multihot_feaslot:
            x = x - 1 + y[1] - y[0]
        return x

    def set_training_mode(self, enable_training):
        pass

    def make_config(self):
        pass

    def init(self, training=False):
        if self.model is not None:
            return
        model_class = self.conf['common']['model_class']
        x = globals()
        if model_class not in x:
            sys.stderr.write("model_class not find: [%s]\n" % (model_class))
            return
        if training:
            self.conf['model_params']['training'] = True
        self.model = x[model_class](**self.conf['model_params'])

    def prepare_dataset(self, ds):
        com_conf = self.conf['common']
        batch_size = com_conf['batch_size']
        shuffle = com_conf['shuffle']
        epoch_num = com_conf['epoch_num']
        if shuffle:
            ds = ds.shuffle(buffer_size=batch_size * 3)
        # if epoch_num > 0:
        #    ds = ds.repeat(epoch_num)
        ds = ds.batch(batch_size).prefetch(batch_size)
        return ds

    @tf.function(experimental_compile=True)
    def train_step(self, label, fea_ids, fea_vals, model):
        with tf.GradientTape() as tape:
            pred = model([fea_ids, fea_vals])
            loss = model.loss(label, pred)
            loss = loss - 0.5 * pred
            gradients = tape.gradient(loss, model.trainable_weights)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return tf.reduce_mean(loss)

    @tf.function(experimental_compile=True)
    def test_step(self, label, fea_ids, fea_vals, model):
        pred = model([fea_ids, fea_vals])
        loss = model.loss(label, pred)
        return tf.reduce_mean(loss, 0), tf.squeeze(pred, 1)

    def train(self, train_data, test_data):
        if self.model is None:
            self.init(training=True)
        com_conf = self.conf['common']
        epoch_num = com_conf['epoch_num']

        model = self.model
        tf.print('start training')
        for epo in range(epoch_num):
            cnt = 0
            pos = 0
            total_loss = []
            for step, feat in enumerate(train_data):
                label, fea_ids, fea_vals = feat
                loss = self.train_step(label, fea_ids, fea_vals, model)
                cnt += label.shape[0]
                pos += sum(label)
                if step % 100 == 9:
                    tf.print(datetime.datetime.now(),
                          'step No.%08d\tloss:%04f, pos: %d, cnt: %d' % (step, loss, pos, cnt))
                total_loss.append(loss)
            total_loss = np.array(total_loss)
            tf.print(datetime.datetime.now(),
                  'step No.%08d\ttotal_loss:%04f, total_pos: %d, total_cnt: %d' % (step, total_loss.mean(), pos, cnt))
            tf.print(datetime.datetime.now(), "epo no:%d finish" % (epo))
            self.set_training_mode(False)
            self.test(test_data)
            self.set_training_mode(True)
            model.save_weights("%s/tfmodel_%d.h5" % (com_conf['local_model_dir'], epo))

        conf = self.conf['common']
        model.save_weights(conf['local_model_dir'] + '/tfmodel.h5')

    def test(self, test_data, model_path=''):
        res = []
        loss_sum = 0.0
        pos = 0
        for step, feat in enumerate(test_data):
            label, fea_ids, fea_vals = feat
            loss, pred = self.test_step(label, fea_ids, fea_vals, self.model)
            loss_sum += loss
            pred = pred.numpy().tolist()
            label = label.numpy().reshape(-1).tolist()
            pos += sum(label)
            res.extend(zip(label, pred))
            if step % 100 == 0:
                print("step :%d loss:%04f, pos: %d" % (step, loss_sum / float(step + 1), pos))
        print(model_path, "test auc:%f size:%d loss:%f, pos: %d" % (ut.auc(res), (step + 1), loss_sum / float(step + 1), pos))
        pass

    @tf.function(experimental_compile=True)
    def test2_step(self, label, fea_ids, fea_vals, model):
        pred = model([fea_ids, fea_vals])
        loss = model.loss(label, pred)
        return tf.reduce_mean(loss, 0), tf.squeeze(pred, 1)

    def test2(self, test_data, model_path=''):
        res = []
        loss_sum = 0.0
        pos = 0
        low_score_num = 0
        for step, feat in test_data.enumerate():
            label, fea_ids, fea_vals, add_info = feat
            loss, pred = self.test2_step(label, fea_ids, fea_vals, self.model)
            loss_sum += loss
            pred = pred.numpy().tolist()
            label = label.numpy().reshape(-1).tolist()
            pos += sum(label)
            add_info = add_info.numpy().reshape(-1).tolist()
            res.extend(zip(label, pred, add_info))
            low_score_num += len(list((filter(lambda x: x < 0.01, pred))))
            if step % 100 == 0:
                print("step :%d loss:%04f, pos: %d" % (step, loss_sum / float(step + 1), pos))
        auc_score, group_auc, u_avg_auc, o_auc_score, o_group_auc, o_u_avg_auc = ut.multi_auc(res)
        auc_score2, group_auc2, u_avg_auc2, o_auc_score2, o_group_auc2, o_u_avg_auc2, valid_count, valid_pos_count = ut.multi_auc_filter_valid_shop(res)
        print(model_path, "low score rate:%f" % (low_score_num / (step + 1)))
        print(model_path, "test auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
        auc_score, group_auc, u_avg_auc, (step + 1), loss_sum / float(step + 1), pos))
        print(model_path, "online auc:%f gauc:%f uauc:%f " % (o_auc_score, o_group_auc, o_u_avg_auc))
        print(model_path,
              "test valid_shop_auc:%f valid_shop_gauc:%f valid_shop_uauc:%f size:%d pos: %d" % (auc_score2, group_auc2, u_avg_auc2, valid_count, valid_pos_count))
        print(model_path,
              "online valid_shop_auc:%f valid_shop_gauc:%f valid_shop_uauc:%f " % (o_auc_score2, o_group_auc2, o_u_avg_auc2))
        pass

    def dump_serving_model(self, model_path):
        if self.model is None:
            self.init()
        self.model.is_save_model = True
        self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=['mae'])
        conf = self.conf['common']
        self.model._set_inputs(tf.keras.Input(shape=(2 * conf['padding_size'],), dtype=tf.dtypes.float32))
        # self.model(tf.keras.Input(shape=(2 * conf['padding_size'],), dtype=tf.dtypes.float32))
        self.model.load_weights(args.model)
        self.model.save(conf['local_model_dir'] + '/tfmodel', save_format='tf')

    def predict(self, X, model_path=None):
        if self.model is None:
            self.init()
            if model_path is not None:
                self.model._set_inputs(X)
                # conf=self.conf['common']
                # self.model._set_inputs(tf.keras.Input(shape=(2*conf['padding_size'],), dtype=tf.dtypes.float32))
                # self.model(X)
                self.model.load_weights(model_path)
        pred = self.model(X)
        return tf.squeeze(pred, 1).numpy().tolist()

    def batch_test_local(self, data_set, model_path, batch_size=1024):
        # init models
        res = []
        lis = model_path.split(',')
        for x in lis:
            md = copy.deepcopy(self)
            md.init()
            md.model._set_inputs([data_set.__iter__().next()[1], data_set.__iter__().next()[2]])
            md.model.load_weights(x)
            md.test(data_set, x)

    def batch_test_local2(self, data_set, model_path, batch_size=1024):
        # init models
        res = []
        lis = model_path.split(',')
        for x in lis:
            md = copy.deepcopy(self)
            md.init()
            md.model._set_inputs([data_set.__iter__().next()[1], data_set.__iter__().next()[2]])
            # md.model([data_set.__iter__().next()[1], data_set.__iter__().next()[2]])
            md.model.load_weights(x)
            md.test2(data_set, x)

    def batch_test(self, infs, model_path, batch_size=10240, ofs=None, is_test=True, output_infos=False):
        indices = []
        fea_index = []
        fea_value = []
        labels = []
        infos = []
        mx = -1
        NR = 0
        pos_num = 0
        res = []
        model_list = []
        lis = model_path.split(',')
        for x in lis:
            model_list.append([copy.deepcopy(self), x])
            res.append([])
        for line in infs:
            NR += 1
            lis = line.split('#', 1)[0].strip(' \n').split(' ')
            info = line.split('#', 1)[1].strip(' \n')
            infos.append(info)
            label = int(lis[0])
            if label > 0:
                pos_num += 1
            labels.append(label)
            for i, x in enumerate(lis[1:]):
                y = x.split(':')
                ind = int(y[0])
                val = float(y[1])
                indices.append([NR - 1, i])
                fea_index.append(ind)
                fea_value.append(val)
                if i > mx:
                    mx = i
            if NR == batch_size:
                X_ind = tf.sparse.to_dense(
                    tf.sparse.SparseTensor(indices=indices, values=fea_index, dense_shape=[NR, mx + 1]))
                X_val = tf.sparse.to_dense(
                    tf.sparse.SparseTensor(indices=indices, values=fea_value, dense_shape=[NR, mx + 1]))
                for i, model in enumerate(model_list):
                    pred = model[0].predict([X_ind, X_val], model[1])
                    res[i].extend(zip(labels, pred, infos))
                indices = []
                fea_index = []
                fea_value = []
                infos = []
                labels = []
                NR = 0

        if NR > 0:
            X_ind = tf.sparse.to_dense(
                tf.sparse.SparseTensor(indices=indices, values=fea_index, dense_shape=[NR, mx + 1]))
            X_val = tf.sparse.to_dense(
                tf.sparse.SparseTensor(indices=indices, values=fea_value, dense_shape=[NR, mx + 1]))
            for i, model in enumerate(model_list):
                pred = model[0].predict([X_ind, X_val], model[1])
                res[i].extend(zip(labels, pred, infos))
        if is_test:
            for i, model in enumerate(model_list):
                print("%s test auc:%f size:%d" % (model[1], ut.auc(res[i]), len(res[i])))
        if ofs is not None:
            for r in res:
                for x in r:
                    if output_infos:
                        print('\t'.join([x[2], str(x[1])]), file=ofs)
                    else:
                        print('\t'.join([x[2], str(x[1]), str(x[0])]), file=ofs)
    
    def batch_test_optim(self, dataset, model_path, ofs=None, is_test=True, output_infos=False):
        res = []
        model_list = []
        lis = model_path.split(',')
        for model_name in lis:
            model_list.append([copy.deepcopy(self), model_name])
            res.append([])
        for step, feat in enumerate(dataset):
            label, fea_ids, fea_vals, add_info = feat
            label = label.numpy().reshape(-1).tolist()
            add_info = add_info.numpy().reshape(-1).tolist()
            for i, model in enumerate(model_list):
                pred = model[0].predict([fea_ids, fea_vals], model[1])
                res[i].extend(zip(label, pred, add_info))
        if is_test:
            for i, model in enumerate(model_list):
                print("%s test auc:%f size:%d" % (model[1], ut.auc(res[i]), len(res[i])))
        if ofs is not None:
            for r in res:
                for x in r:
                    if output_infos:
                        print('\t'.join([x[2], str(x[1])]), file=ofs)
                    else:
                        print('\t'.join([x[2].decode(), str(x[1]), str(x[0])]), file=ofs)

class DeepFMLearner(Learner):
    def __init__(self, conf_yml='dfm.yml', conf=None):
        super().__init__(conf_yml, conf)

    def make_config(self):
        if 'model_params' not in self.conf:
            print("please config model_params in %s" % (self.conf_yml), file=sys.stderr)
            return
        fea_slot, feature_size = self.get_fea_slot()
        multihot_fea = self.get_multihot_feaslot(fea_slot)
        multi_fea = self.get_multi_feaslot(fea_slot)
        preemb_fea = self.get_preemb_feaslot(fea_slot)
        x = self.conf['model_params']
        x['feature_size'] = feature_size
        x['field_size'] = len(fea_slot)
        x['multihot_fea'] = multihot_fea
        x['preemb_fea'] = preemb_fea
        x['multi_fea'] = multi_fea
        if self.conf['switch']['enable_calc_padding']:
            self.conf['common']['padding_size'] = self.get_padding_size(fea_slot, multi_fea,
                                                                        self.conf['common']['fix_addition_pad'])
        ut.save_yaml_file(self.conf, self.conf_yml)
        pass

    def set_training_mode(self, enable_training):
        self.model.training = enable_training
        pass


class XDeepFMLearner(Learner):
    def __init__(self, conf_yml="xdfs.yml", conf=None):
        super(XDeepFMLearner, self).__init__(conf_yml, conf)

    def make_config(self):
        if "model_params" not in self.conf:
            # print("please config model_params in %s"%self.conf_yml, file=sys.sys.stderr)
            print("please config model_params in %s" % self.conf_yml)
            return
        fea_slot, feature_size = self.get_fea_slot()
        multi_fea = self.get_multihot_feaslot(fea_slot)
        preemb_fea = self.get_preemb_feaslot(fea_slot)
        x = self.conf["model_params"]
        x["feature_size"] = feature_size
        x["field_size"] = len(fea_slot)
        x["multi_fea"] = multi_fea
        x["preemb_fea"] = preemb_fea
        if self.conf["switch"]["enable_calc_padding"]:
            self.conf["common"]["padding_size"] = self.get_padding_size(fea_slot, multi_fea,
                                                                        self.conf["common"]["fix_addition_pad"])
        ut.save_yaml_file(self.conf, self.conf_yml)
        pass

    def set_training_mode(self, enable_training):
        self.model.training = enable_training
        pass


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='train&evaluate the DeepFM model')
    parse.add_argument('-m', type=str, help='-method:train|test|auto_config', required=True)
    parse.add_argument('-conf', type=str, help='yml conf')
    parse.add_argument('-conf_models', type=str, help='conf1,model1,model2...;conf2,model1,model2...')
    parse.add_argument('-model', type=str, help='init model dir')
    parse.add_argument('-data', type=str, help='input data files')
    parse.add_argument('-dataTrain', type=str, help='input train data files')
    parse.add_argument('-dataTest', type=str, help='input test data files')
    parse.add_argument('-dataOut', type=str, help='out test data file')
    args = parse.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    if args.conf:
        x = ut.load_yaml_file(args.conf)
        mc = x['common']['model_class']
        mcif = mc + "Learner"
        z = globals()
        solver = None
        if mcif in z:
            solver = z[mcif](args.conf, x)
        else:
            solver = Learner(args.conf, x)
            print("no model interface %s find" % (mcif), file=sys.stderr)
        # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        # print(gpus)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(x['common']['CUDA_VISIBLE_DEVICES'])
        # 新增：确保 GPU 内核能够从自己的专用线程启动
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.m == 'train':
        info = {}
        com_conf = x['common']
        batch_size = com_conf['batch_size']
        shuffle_size = com_conf['shuffle_size']
        preprocess_conf = com_conf['preprocess_conf']
        train_ds = ut.ReadTextLineDatasetOnline(args.dataTrain, conf_file=preprocess_conf, shuffle_size=shuffle_size,
                                               batch_size=batch_size, fetch_size=tf.data.experimental.AUTOTUNE, num_parallel=tf.data.experimental.AUTOTUNE)
        test_ds = ut.ReadTextLineDatasetOnline(args.dataTest, conf_file=preprocess_conf, shuffle_size=shuffle_size,
                                                batch_size=batch_size, fetch_size=tf.data.experimental.AUTOTUNE, num_parallel=tf.data.experimental.AUTOTUNE)
        # ds = ut.svmlight2dataset(sys.stdin, info)
        # print("total ins num:",info['all_num'])
        # print("postive ins num:",info['pos_num'])
        solver.train(train_ds, test_ds)
    elif args.m == 'test_xxx':
        solver.init()
        info = {}
        ds = ut.svmlight2dataset(sys.stdin, info)
        print("total ins num:", info['all_num'])
        print("postive ins num:", info['pos_num'])
        ds = ds.batch(1024)
        # 1: fea_ids, 2: fea_vals. feat: (label, fea_id, fea_val)
        solver.model._set_inputs([ds.__iter__().next()['1'], ds.__iter__().next()[2]])
        solver.model.load_weights(args.model)
        solver.test(ds)
        pass
    elif args.m == 'test':
        com_conf = x['common']
        batch_size = com_conf['batch_size']
        preprocess_conf = com_conf['preprocess_conf']
        if args.data:
            ds = ut.ReadTextLineDatasetOnline(args.data, conf_file=preprocess_conf, batch_size=batch_size,
                                              fetch_size=tf.data.experimental.AUTOTUNE,num_parallel=tf.data.experimental.AUTOTUNE)
            solver.batch_test_local(ds, args.model)
        else:
            solver.batch_test(sys.stdin, args.model)

    elif args.m == 'test_diff_nets':
        nets = args.conf_models
        for net in nets.split(';'):
            conf, model = net.split(',', 1)
            x = ut.load_yaml_file(conf)
            com_conf = x['common']
            preprocess_conf = com_conf['preprocess_conf']
            batch_size = com_conf['batch_size']
            ds = ut.ReadTextLineDatasetOnline(args.data, conf_file=preprocess_conf, batch_size=batch_size, fetch_size=1,
                                              num_parallel=10)
            solver = Learner(conf, ut.load_yaml_file(conf))
            solver.batch_test_local(ds, model)
    elif args.m == 'test_diff_nets2':
        nets = args.conf_models
        for net in nets.split(';'):
            conf, model = net.split(',', 1)
            x = ut.load_yaml_file(conf)
            com_conf = x['common']
            preprocess_conf = com_conf['preprocess_conf']
            batch_size = com_conf['test_batch_size']
            print("start test_diff_nets2: batch_size is ", batch_size)
            ds = ut.ReadTextLineDatasetOnlineAddInfo(args.data, conf_file=preprocess_conf, batch_size=batch_size,
                                                     fetch_size=tf.data.experimental.AUTOTUNE, num_parallel=tf.data.experimental.AUTOTUNE)
            solver = Learner(conf, ut.load_yaml_file(conf))
            solver.batch_test_local2(ds, model)

    elif args.m == 'predict':
        test_batch_size = x['common']['test_batch_size']
        print("start predict: test_batch_size is ", test_batch_size)
        if args.dataOut:
            f = open(args.dataOut, 'w')
            solver.batch_test(sys.stdin, args.model, ofs=f, batch_size=test_batch_size)
            f.close()
        else:
            solver.batch_test(sys.stdin, args.model, ofs=sys.stdout, batch_size=test_batch_size)
    elif args.m == "predict_optim":
        test_batch_size = x['common']['test_batch_size']
        preprocess_conf = x['common']["preprocess_conf"]
        print("start predict: test_batch_size is ", test_batch_size)
        if args.data:
            ds = ut.ReadTextLineDatasetOnlineAddInfo(args.data, conf_file=preprocess_conf, batch_size=test_batch_size,
                                              fetch_size=tf.data.experimental.AUTOTUNE,
                                              num_parallel=tf.data.experimental.AUTOTUNE)
        if args.dataOut:
            f = open(args.dataOut, 'w')
            solver.batch_test_optim(ds, args.model, ofs=f)
            f.close()
        else:
            solver.batch_test_optim(ds, args.model, ofs=sys.stdout)
    elif args.m == 'predict_info':
        solver.batch_test(sys.stdin, args.model, ofs=sys.stdout, output_infos=True)
    elif args.m == 'dump_serving':
        solver.dump_serving_model(args.model)
    elif args.m == 'auto_config':
        solver.make_config()
    else:
        pass
