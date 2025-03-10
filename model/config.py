# Loss
lb = 1.
lb_mask = 1.
lb_beta = 5.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
lf_mask = 10.
lf_rec = 0.1

# StyleAug
vflip_rate = 0.5
hflip_rate = 0.5
angle_range = [(-15, -5), (5, 15)]

# Train
learning_rate = 1e-4
decay_rate = 0.9
beta1 = 0.9
beta2 = 0.999
max_iter = 30000
write_log_interval = 5000
save_ckpt_interval = 5000
gen_example_interval = 5000
task_name = 'full-train-TPS'
checkpoint_savedir = 'output/' + task_name + '/'  # dont forget '/'
ckpt_path = 'output/full-train-TPS/train_step-105000.model' #'output/full-train/train_step-85000.model'
inpaint_ckpt_path = ''#'output/erase-train/new_data_mean/train_step-100000.model'
vgg19_weights = 'models/vgg19-dcbb9e9d.pth'

# data
batch_size = 2
real_bs = 0
with_real_data = True if real_bs > 0 else False
data_shape = [128, 128]
num_workers = 1
data_dir = '../syn_datagen/SRNet-Datagen/result'
i_s_dir = 'i_s'
t_b_dir = 't_b'
i_t_dir = 'i_t'
t_f_dir = 't_f'
mask_s_dir = 'mask_s'
mask_t_dir = 'mask_2'
example_data_dir = '../test'
example_result_dir = checkpoint_savedir + 'val_visualization'

# TPS
TPS_ON = True
num_control_points = 10
stn_activation = 'tanh'
tps_inputsize = data_shape
tps_outputsize = data_shape
tps_margins = (0.05, 0.05)

# predict
predict_ckpt_path = None
predict_data_dir = None
predict_result_dir = checkpoint_savedir + 'pred_result'
