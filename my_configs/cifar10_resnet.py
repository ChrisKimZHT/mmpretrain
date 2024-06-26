_base_ = [
    '../configs/_base_/default_runtime.py',
    '../configs/_base_/datasets/cifar10_bs16.py',
    '../configs/_base_/schedules/cifar10_bs128.py',
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

pipeline = [dict(type='PackInputs')]

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

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,
                    max_keep_ckpts=20, save_best="auto"),
)