custom_imports = dict(
    imports=['datasets.tree_canopy'],
    allow_failed_imports=False
)

_base_ = [
    'C:/Users/harsh/Desktop/Keinisha/mmsegmentation/configs/_base_/models/unet_s5-d16.py',
    'C:/Users/harsh/Desktop/Keinisha/mmsegmentation/configs/_base_/default_runtime.py',
    'C:/Users/harsh/Desktop/Keinisha/mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

# ---------- Dataset ----------
dataset_type = 'TCD_Dataset'
data_root = 'data/training_data_object_detection/'

classes = ('background', 'individual_tree', 'group_of_trees')
palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)
# ---------- Pipelines ----------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=16, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# ---------- Dataloader ----------
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='mmseg_masks/train',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='mmseg_masks/val',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        test_mode=False
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='mmseg_masks/val',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        test_mode=True
    ),
)

# ---------- Model ----------
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=34,
        in_channels=3,
        num_stages=4,
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 2),
        out_indices=(3,),
        style='pytorch'
    ),
    decode_head=dict(
        type='UNetHead',
        in_channels=512,
        channels=256,
        num_classes=3,           # <<< 3 classes
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=0.7),
            dict(type='DiceLoss', loss_weight=0.3)
        ]
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ---------- FP16 ----------
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 = dict()
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)

# ---------- Runner (MMSeg 0.x) ----------
runner = dict(type='IterBasedRunner', max_iters=40000)

# ---------- Hooks ----------
checkpoint_config = dict(by_epoch=False, interval=2000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# ---------- Evaluation ----------
evaluation = dict(interval=5000, metric='mIoU')