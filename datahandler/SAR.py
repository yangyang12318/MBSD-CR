
from datahandler.denoise_dataset import DenoiseDataSet
import os




class SARdata(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    #QXSLAB_SAROPT
    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'train')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'sar')):
            self.img_paths = files

    def _load_data(self, data_idx):

        file_name = self.img_paths[data_idx]

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'sar', file_name), as_gray=True)
        NLsar = self._load_img(os.path.join(self.dataset_path, 'NLsar', file_name), as_gray=True)

        return {'real_noisy': noisy_img,'NLsar':NLsar,'file_name':file_name}


class SARdataval(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    #QXSLAB_SAROPT
    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'test')  # 这里先用测试数据，记得修改回来
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'sar')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'sar', file_name), as_gray=True)
        NLsar = self._load_img(os.path.join(self.dataset_path, 'NLsar', file_name), as_gray=True)

        return {'real_noisy': noisy_img, 'NLsar': NLsar, 'file_name': file_name}


class SARdatatest(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #QXSLAB_SAROPT
    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'xinjiang')  # 这里先用测试数据，记得修改回来
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'NLs1')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, 's1', file_name), as_gray=True)
        NLsar = self._load_img(os.path.join(self.dataset_path, 'NLs1', file_name), as_gray=True)
        return {'real_noisy': noisy_img, 'NLsar': NLsar, 'file_name': file_name}
