import yaml


class Config:
    def __init__(self):
        self.load_model_epoch = None
        self.data_loader_seed = None
        self.n_epochs = None

        self.debug = None
        self.total_epochs = None
        self.warmup_epochs = None
        self.image_size = None
        self.model_img_save_epoch = None

        self.batch_size = None
        self.n_split = None
        self.d_max_lr = None
        self.d_min_lr = None
        self.g_max_lr = None
        self.g_min_lr = None
        self.lambda_cycle = None
        self.seed = None
        self.root_dir = None

        self.domain_a_dir = None
        self.domain_b_dir = None

        self.plot_color_pathes = None
        self.plot_hist_pathes = None
        self.load()

    def load(self):
        with open('/pokemon/config/config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

            self.load_model_epoch = config.get('LOAD_MODEL_EPOCH')
            self.data_loader_seed = config.get('DATA_LOADER_SEED')
            self.n_epochs = config.get('N_EPOCHS')

            self.debug = config.get('DEBUG')
            self.total_epochs = config.get('TOTAL_EPOCHS')
            self.warmup_epochs = config.get('WARMUP_EPOCHS')
            self.image_size = config.get('IMAGE_SIZE')
            self.model_img_save_epoch = config.get('MODEL_IMG_SAVE_EPOCH')

            self.batch_size = config.get('BATCH_SIZE')
            self.n_split = config.get('N_SPLIT')
            self.d_max_lr = config.get('D_MAX_LR')
            self.d_min_lr = config.get('D_MIN_LR')
            self.g_max_lr = config.get('G_MAX_LR')
            self.g_min_lr = config.get('G_MIN_LR')
            self.lambda_cycle = config.get('LAMBDA_CYCLE')
            self.seed = config.get('SEED')
            self.root_dir = config.get('ROOT_DIR')

            self.domain_a_dir = config.get('DOMAIN_A_DIR')
            self.domain_b_dir = config.get('DOMAIN_B_DIR')

            self.plot_color_pathes = config.get('PLOT_COLOR_PATHES')
            self.plot_hist_pathes = config.get('PLOT_HIST_PATHES')
