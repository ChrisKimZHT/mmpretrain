_base_ = [
    '../configs/_base_/default_runtime.py',
    '../configs/_base_/datasets/cifar10_bs16.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

pipeline = [
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        data_root='/home/chriskim/cifar-10-python',
        pipeline=pipeline
    ),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        data_root='/home/chriskim/cifar-10-python',
        pipeline=pipeline
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    ),
)

param_scheduler = dict(type='MultiStepLR', by_epoch=True,
                       milestones=[30, 60, 90], gamma=0.1)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

test_cfg = dict()

val_cfg = dict()

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,
                    max_keep_ckpts=20, save_best="auto"),
)

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
)
