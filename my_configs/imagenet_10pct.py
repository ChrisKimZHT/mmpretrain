data_preprocessor = dict(
    num_classes=100,
    mean=[123.7284, 117.4751, 105.1077],
    std=[57.4626, 56.4878, 56.9013],
    to_rgb=True,
)

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CustomDataset',
        data_root='/home/chriskim/ILSVRC2012',
        data_prefix='train',
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CustomDataset',
        data_root='/home/chriskim/ILSVRC2012',
        data_prefix='val',
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)
