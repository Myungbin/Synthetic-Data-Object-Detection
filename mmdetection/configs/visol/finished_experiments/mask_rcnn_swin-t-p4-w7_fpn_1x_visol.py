_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py'
    ]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

dataset_type = 'CarDataset'
data_root = r'C:\MB_Project\project\Competition\VISOL\data\\'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='MixUp', img_scale=(1333, 800), ratio_range=(0.8, 1.6), pad_val=144.0),
    dict(type='Corrupt', corruption='fog'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CarDataset',
            ann_file=r'/data/train.txt',
            img_prefix=r'C:\MB_Project\project\Competition\VISOL\data',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ]),
        pipeline=train_pipeline),
    val=dict(
        type='CarDataset',
        test_mode=False,
        ann_file=r'/data/val.txt',
        img_prefix=r'C:\MB_Project\project\Competition\VISOL\data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='MultiScaleFlipAug', img_scale=(1333, 800), flip=False,
                 transforms=[
                     dict(type='Resize', keep_ratio=True),
                     dict(type='RandomFlip'),
                     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                     dict(type='Pad', size_divisor=32),
                     dict(type='ImageToTensor', keys=['img']),
                     dict(type='Collect', keys=['img'])
                 ])
        ]),
    test=dict(
        type='CarDataset',
        ann_file=r'/data/test.txt',
        img_prefix=r'C:\MB_Project\project\Competition\VISOL\data',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='MultiScaleFlipAug', img_scale=(1333, 800), flip=False,
                 transforms=[
                     dict(type='Resize', keep_ratio=True),
                     dict(type='RandomFlip'),
                     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                     dict(type='Pad', size_divisor=32),
                     dict(type='ImageToTensor', keys=['img']),
                     dict(type='Collect', keys=['img'])
                 ])
        ]))

evaluation = dict(interval=1, metric='mAP', iou_thr=0.85)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=16)
work_dir = r'/configs/visol'
auto_resume = False
gpu_ids = [0]

