data_preprocessor = dict(
    num_classes=120,
    mean=[118.528, 116.1984, 110.4128],
    std=[58.4448, 57.0368, 61.9776],
    to_rgb=True,
)

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=384),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CustomDataset',
        data_root='/home/chriskim/wuhan-scenery-selected',
        data_prefix='train',
        pipeline=pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CustomDataset',
        data_root='/home/chriskim/wuhan-scenery-selected',
        data_prefix='val',
        pipeline=pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)
