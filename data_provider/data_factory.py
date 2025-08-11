from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'electricity': Dataset_Custom,
    'electricity1': Dataset_Custom,
    'electricity2': Dataset_Custom,
    'electricity3': Dataset_Custom,
    'electricity_1': Dataset_Custom,
    'electricity_2': Dataset_Custom,
    'electricity_3': Dataset_Custom,
    'electricity_4': Dataset_Custom,
    'electricity_5': Dataset_Custom,
    'electricity_6': Dataset_Custom,
    'electricity_7': Dataset_Custom,
    'traffic': Dataset_Custom,
    'traffic1': Dataset_Custom,
    'traffic2': Dataset_Custom,
    'traffic3': Dataset_Custom,
    'traffic4': Dataset_Custom,
    'traffic5': Dataset_Custom,
    'traffic6': Dataset_Custom,
    'traffic7': Dataset_Custom,
    'traffic8': Dataset_Custom,
    'traffic9': Dataset_Custom,
    'traffic_1': Dataset_Custom,
    'traffic_2': Dataset_Custom,
    'traffic_3': Dataset_Custom,
    'traffic_4': Dataset_Custom,
    'traffic_5': Dataset_Custom,
    'traffic_6': Dataset_Custom,
    'traffic_7': Dataset_Custom,
    'traffic_8': Dataset_Custom,
    'traffic_9': Dataset_Custom,
    'traffic_10': Dataset_Custom,
    'traffic_11': Dataset_Custom,
    'traffic_12': Dataset_Custom,
    'traffic_13': Dataset_Custom,
    'traffic_14': Dataset_Custom,
    'traffic_15': Dataset_Custom,
    'traffic_16': Dataset_Custom,
    'traffic_17': Dataset_Custom,
    'traffic_18': Dataset_Custom,
    'traffic_OT_0': Dataset_Custom,
    'traffic_OT_1': Dataset_Custom,
    'traffic_OT_2': Dataset_Custom,
    'traffic_OT_3': Dataset_Custom,
    'traffic_OT_4': Dataset_Custom,
    'traffic_OT_5': Dataset_Custom,
    'traffic_OT_6': Dataset_Custom,
    'traffic_OT_7': Dataset_Custom,
    'traffic_OT_8': Dataset_Custom,
    'traffic_OT_9': Dataset_Custom,
    'traffic_OT_10': Dataset_Custom,
    'traffic_OT_11': Dataset_Custom,
    'traffic_OT_12': Dataset_Custom,
    'traffic_OT_13': Dataset_Custom,
    'traffic_OT_14': Dataset_Custom,
    'traffic_OT_15': Dataset_Custom,
    'traffic_OT_16': Dataset_Custom,
    'traffic_OT_17': Dataset_Custom,
    'traffic_OT_18': Dataset_Custom,
    'traffic_OT_19': Dataset_Custom,
    'traffic_OT_20': Dataset_Custom,
    'traffic_OT_21': Dataset_Custom,
    'traffic_OT_21': Dataset_Custom,
    'traffic_OT_22': Dataset_Custom,
    'traffic_OT_23': Dataset_Custom,
    'traffic_OT_24': Dataset_Custom,
    'traffic_OT_25': Dataset_Custom,
    'traffic_OT_26': Dataset_Custom,
    'traffic_OT_27': Dataset_Custom,
    'traffic_OT_28': Dataset_Custom,
    'traffic_OT_29': Dataset_Custom,
    'traffic_OT_30': Dataset_Custom,
    'traffic_OT_31': Dataset_Custom,
    'traffic_OT_32': Dataset_Custom,
    'traffic_OT_33': Dataset_Custom,
    'traffic_OT_34': Dataset_Custom,
    'traffic_OT_35': Dataset_Custom,
    'traffic_OT_36': Dataset_Custom,
    'traffic_OT_37': Dataset_Custom,
    'traffic_OT_38': Dataset_Custom,
    'traffic_OT_39': Dataset_Custom,
    'traffic_OT_40': Dataset_Custom,
    'traffic_OT_40': Dataset_Custom,
    'traffic_OT_41': Dataset_Custom,
    'traffic_OT_42': Dataset_Custom,
    'traffic_OT_43': Dataset_Custom,
    'traffic_OT_44': Dataset_Custom,
    'traffic_OT_45': Dataset_Custom,
    'traffic_OT_46': Dataset_Custom,
    'traffic_OT_47': Dataset_Custom,
    'traffic_OT_48': Dataset_Custom,
    'traffic_OT_49': Dataset_Custom,
    'traffic_OT_50': Dataset_Custom,
    'weather': Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'national_illness': Dataset_Custom,
    'kospi_original': Dataset_Custom,
    'kospi_original_test': Dataset_Custom,
    'kospi_multi_test2': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            if args.model == 'TimesNet':
                batch_size = 1  # bsz=1 for evaluation
            else:
                batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns, train_ratio=args.train_ratio
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
