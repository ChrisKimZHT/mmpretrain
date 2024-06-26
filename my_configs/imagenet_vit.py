_base_ = [
    '../configs/_base_/default_runtime.py',
    '../configs/_base_/schedules/imagenet_bs4096_AdamW.py',
    './imagenet_10pct.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=100,
        in_channels=768,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1,
                  mode='classy_vision'),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ])
)

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/home/chriskim/ILSVRC2012',
        pipeline=pipeline
    ),
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/home/chriskim/ILSVRC2012',
        pipeline=pipeline
    ),
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))

train_cfg = dict(max_epochs=100)

test_cfg = None

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
