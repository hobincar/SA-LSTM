import os
import time


class MSVDSplitConfig:
    model = "MSVD_ResNet152"

    video_fpath = "data/MSVD/features/{}.hdf5".format(model)
    caption_fpath = "data/MSVD/metadata/MSR Video Description Corpus.csv"

    train_video_fpath = "data/MSVD/features/{}_train.hdf5".format(model)
    val_video_fpath = "data/MSVD/features/{}_val.hdf5".format(model)
    test_video_fpath = "data/MSVD/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "data/MSVD/metadata/train.csv"
    val_metadata_fpath = "data/MSVD/metadata/val.csv"
    test_metadata_fpath = "data/MSVD/metadata/test.csv"


class MSRVTTSplitConfig:
    model = "MSR-VTT-v2016_ResNet152"

    video_fpath = "data/MSR-VTT/features/{}.hdf5".format(model)
    train_val_caption_fpath = "data/MSR-VTT/metadata/train_val_videodatainfo.json"
    test_caption_fpath = "data/MSR-VTT/metadata/test_videodatainfo.json"

    train_video_fpath = "data/MSR-VTT/features/{}_train.hdf5".format(model)
    val_video_fpath = "data/MSR-VTT/features/{}_val.hdf5".format(model)
    test_video_fpath = "data/MSR-VTT/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "data/MSR-VTT/metadata/train.json"
    val_metadata_fpath = "data/MSR-VTT/metadata/val.json"
    test_metadata_fpath = "data/MSR-VTT/metadata/test.json"
    total_metadata_fpath = "data/MSR-VTT/metadata/total.json"


class FeatureConfig:
    models = [ "MSVD_ResNet152" ]
    size = 0
    for model in models:
        if 'AlexNet' in model:
            size += 9216
        elif 'VGG' in model or 'C3D' in model:
            size += 4096
        elif 'ResNet' in model or 'ResNext' in model or 'InceptionV3' in model:
            size += 2048
        elif 'DenseNet' in model:
            size += 1920
        elif 'InceptionV4' in model:
            size += 1536
        elif 'ShuffleNet' in model or 'GoogleNet' in model:
            size += 1024
        elif 'R2.5D' in model:
            size += 512
        else:
            raise NotImplementedError("Unknown model: {}".format(model))


class VocabConfig:
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2 }
    embedding_size = 468


class MSVDLoaderConfig:
    n_train = 1200
    n_val = 100
    n_test = 670

    total_caption_fpath = "data/MSVD/metadata/MSR Video Description Corpus.csv"
    train_caption_fpath = "data/MSVD/metadata/train.csv"
    val_caption_fpath = "data/MSVD/metadata/val.csv"
    test_caption_fpath = "data/MSVD/metadata/test.csv"
    min_count = 1
    max_caption_len = 30

    total_video_feat_fpath_tpl = "data/{}/features/{}.hdf5"
    phase_video_feat_fpath_tpl = "data/{}/features/{}_{}.hdf5"
    frame_sampling_method = 'uniform'; assert frame_sampling_method in [ 'uniform', 'random' ]
    frame_max_len = 300 // 5
    frame_sample_len = 28

    num_workers = 4


class MSRVTTLoaderConfig:
    n_train = 5175
    n_val = 398
    n_test = 2354

    total_caption_fpath = "data/MSR-VTT/metadata/total.json"
    train_caption_fpath = "data/MSR-VTT/metadata/train.json"
    val_caption_fpath = "data/MSR-VTT/metadata/val.json"
    test_caption_fpath = "data/MSR-VTT/metadata/test.json"
    min_count = 1
    max_caption_len = 30

    total_video_feat_fpath_tpl = "data/{}/features/{}.hdf5"
    phase_video_feat_fpath_tpl = "data/{}/features/{}_{}.hdf5"
    frame_sampling_method = 'uniform'; assert frame_sampling_method in [ 'uniform', 'random' ]
    frame_max_len = 300 // 5
    frame_sample_len = 28

    num_workers = 4


class DecoderConfig:
    rnn_type = 'LSTM'; assert rnn_type in [ 'LSTM', 'GRU' ]
    rnn_num_layers = 1
    rnn_num_directions = 1; assert rnn_num_directions in [ 1, 2 ]
    rnn_hidden_size = 512
    rnn_attn_size = 256
    rnn_dropout = 0.5
    rnn_teacher_forcing_ratio = 1.0


class TrainConfig:
    corpus = 'MSVD'; assert corpus in [ 'MSVD', 'MSR-VTT' ]

    feat = FeatureConfig
    vocab = VocabConfig
    loader = {
        'MSVD': MSVDLoaderConfig,
        'MSR-VTT': MSRVTTLoaderConfig
    }[corpus]
    decoder = DecoderConfig


    """ Optimization """
    epochs = {
        'MSVD': 50,
        'MSR-VTT': 30,
    }[corpus]
    batch_size = 200
    shuffle = True
    optimizer = "AMSGrad"
    gradient_clip = 5.0 # None if not used
    lr = {
        'MSVD': 5e-5,
        'MSR-VTT': 2e-4,
    }[corpus]
    lr_decay_start_from = 20
    lr_decay_gamma = 0.5
    lr_decay_patience = 5
    weight_decay = 1e-5
    reg_lambda = 0.

    """ Pretrained Model """
    pretrained_decoder_fpath = None

    """ Evaluate """
    metrics = [ 'Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L' ]

    """ ID """
    exp_id = "SA-LSTM"
    feat_id = "FEAT {} mfl-{} fsl-{} mcl-{}".format('+'.join(feat.models), loader.frame_max_len, loader.frame_sample_len,
                                                    loader.max_caption_len)
    embedding_id = "EMB {}".format(vocab.embedding_size)
    decoder_id = "DEC {}-{}-l{}-h{} at-{}".format(
        ["uni", "bi"][decoder.rnn_num_directions-1], decoder.rnn_type,
        decoder.rnn_num_layers, decoder.rnn_hidden_size, decoder.rnn_attn_size)
    optimizer_id = "OPTIM {} lr-{}-dc-{}-{}-{}-wd-{} rg-{}".format(
        optimizer, lr, lr_decay_start_from, lr_decay_gamma, lr_decay_patience, weight_decay, reg_lambda)
    hyperparams_id = "bs-{}".format(batch_size)
    if gradient_clip is not None:
        hyperparams_id += " gc-{}".format(gradient_clip)
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    model_id = " | ".join([ exp_id, corpus, feat_id, embedding_id, decoder_id, optimizer_id, timestamp ])

    """ Log """
    log_dpath = "logs/{}".format(model_id)
    ckpt_dpath = os.path.join("checkpoints", model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    save_from = 1
    save_every = 1

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_cross_entropy_loss = "loss/train/decoder_CE"
    tx_train_entropy_loss = "loss/train/decoder_reg"
    tx_val_loss = "loss/val"
    tx_val_cross_entropy_loss = "loss/val/decoder_CE"
    tx_val_entropy_loss = "loss/val/decoder_reg"
    tx_lr = "params/decoder_LR"


class EvalConfig:
    ckpt_fpath = "checkpoints/SA-LSTM | MSR-VTT | FEAT InceptionV4 mcl-30 | EMB 468 | DEC uni-LSTM-l1-h512 at-256 | OPTIM AMSGrad lr-0.0002-dc-20-0.9-5-wd-1e-05 rg-0.001 | 190307-19:10:55/35.ckpt"
    result_dpath = "results"

