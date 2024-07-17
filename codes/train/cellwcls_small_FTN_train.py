_base_=['../queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py']
num_stages=6
model = dict(type='QueryInst',roi_head=dict(type='SparseRoIHead',
            bbox_head=[
            dict(
                type='DIIHead',
                num_classes=3,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ],
        mask_head=[
            dict(
                type='DynamicMaskHead',
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=14,
                    with_proj=False,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                num_convs=4,
                num_classes=3,
                roi_feat_size=14,
                in_channels=256,
                conv_kernel_size=3,
                conv_out_channels=256,
                class_agnostic=False,
                norm_cfg=dict(type='BN'),
                upsample_cfg=dict(type='deconv', scale_factor=2),
                loss_mask=dict(
                    type='DiceLoss',
                    loss_weight=8.0,
                    use_sigmoid=True,
                    activate=False,
                    eps=1e-5)) for _ in range(num_stages)
        ]))
classes = ('single_cell','spanning_cell','empty_cell',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(90, 700), (122, 700), (154, 700), (186, 700),
                           (218, 700), (250, 700), (282, 700), (314, 700),
                           (346, 700), (378, 700), (410, 700)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(200, 700), (300, 700), (400, 700)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(90, 700), (122, 700), (154, 700), (186, 700),
                           (218, 700), (250, 700), (282, 700), (314, 700),
                           (346, 700), (378, 700), (410, 700)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(200, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,

    train=dict(
           ann_file='/share/home/pangliucheng/lc/project_table/dataset/GridCellLabel/FTN/FTNcellboxwcls_train.json',
            img_prefix='/share/home/pangliucheng/lc/project_table/dataset/FinTabNet/train/',
            classes = classes,
        pipeline=train_pipeline
        ),
    #/data/lcpang/lc/project_table/dataset/Pubtables-1M/Pubtables/val
       # test=dict(ann_file='/data/lcpang/lc/project_table/dataset/TableBank_Detection/annotations/tablebank_word_test.json',img_prefix='/data/lcpang/lc/project_table/dataset/TableBank_Detection/images/',classes = classes), 
   test=dict( ann_file='/share/home/pangliucheng/lc/project_table/dataset/GridCellLabel/FTN/FTNcellboxwcls_test.json',
            img_prefix='/share/home/pangliucheng/lc/project_table/dataset/FinTabNet/test/',
            classes = classes,
        pipeline=test_pipeline
            ),
        
        val=dict(
             ann_file='/share/home/pangliucheng/lc/project_table/dataset/GridCellLabel/FTN/FTNcellboxwcls_test.json',
            img_prefix='/share/home/pangliucheng/lc/project_table/dataset/FinTabNet/test/',
            classes = classes,
        pipeline=test_pipeline
        )
        
)
load_from = '/share/home/pangliucheng/lc/project_table/work_dirs/cellwlcs_queryinst_small_FTN_train/resume5e-6/epoch_11(best).pth'
# 以PubTables-1M为训练集的best epoch=9 为模型初始化参数 2.5e-4为初始学习率 ##存到resume2.5e-4中
## 以FTN epoch=3 调低学习率0.1被接着训。
#resume_from = '/share/home/pangliucheng/lc/project_table/models/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth'
resume = False
runner = dict(type = 'EpochBasedRunner',max_epochs=24) 
optimizer = dict(type='AdamW', lr=5e-6,weight_decay=0.0001) #以2.5e-4为初始学习率,计划每隔2轮降低0.1倍.
##调低为2.5e-5
###调低为1e-5
####调低为5e-6
log_config=dict (interval= 500, hooks=[dict(type='TextLoggerHook')])


checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'segm'],interval=3)
auto_scale_lr = dict(enable=False, base_batch_size=16)

work_dir = './work_dirs/cellwlcs_queryinst_small_FTN_train/'
gpu_ids = range(0, 2)