import re
import argparse
from pathlib import Path

from scipy import stats

from utils.data_utilities import *
from utils.SELD_metrics import SELDMetrics


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    使用jackknife方法计算统计量
    
    参数:
        global_value: 使用所有(N个)样本计算得到的全局值
        partial_estimates: 每次使用N-1个样本计算得到的部分估计值
        significance_level: 用于t检验的显著性水平
        
    返回:
        estimate: 使用部分估计值计算得到的估计值
        bias: 全局值与部分估计值之间的偏差
        std_err: 部分估计值的标准差
        conf_interval: t检验后得到的置信区间
    """

    # 计算部分估计值的均值
    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    # 计算偏差
    bias = (n - 1) * (mean_jack_stat - global_value)

    # 计算标准差
    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # 计算偏差校正后的"jackknife估计值"
    estimate = global_value - bias

    # 计算jackknife置信区间
    if not (0 < significance_level < 1):
        raise ValueError("置信水平必须在(0, 1)之间")

    # 获取t值
    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # 执行t检验
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


class ComputeSELDResults(object):
    def __init__(
            self, ref_files_folder=None, use_polar_format=True, average='micro', doa_thresh=20, nb_classes=12, 
            fast_adaptation=False, num_support_samples=0, 
    ):
        """
        初始化SELD结果计算类
        
        参数:
            ref_files_folder: 参考文件文件夹路径
            use_polar_format: 是否使用极坐标格式
            average: 平均方式('micro'或'macro')
            doa_thresh: 到达方向(DOA)阈值
            nb_classes: 类别数量
            fast_adaptation: 是否使用快速适应
            num_support_samples: 支持样本数量
        """
        self._use_polar_format = use_polar_format
        self._desc_dir = Path(ref_files_folder)
        self._doa_thresh = doa_thresh
        self._nb_classes = nb_classes
        self._fast_adaptation = fast_adaptation
        self._num_support_samples = num_support_samples * 10

        # 收集参考文件
        self._ref_meta_list = sorted(self._desc_dir.glob('**/*.csv'))
        self._ref_labels = {}
        room = None
        for file in self._ref_meta_list:
            fn = file.stem
            gt_dict = load_output_format_file(file)
            nb_ref_frames = max(list(gt_dict.keys()))
            
            self._ref_labels[fn] = [to_metrics_format(gt_dict, nb_ref_frames, label_resolution=0.1), nb_ref_frames, gt_dict]

        self._nb_ref_files = len(self._ref_labels)
        self._average = average

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        根据标签获取文件子集
        
        支持的标签:
        'all' - 所有文件
        'ir' - 脉冲响应文件
        
        参数:
            file_list: 预测文件的完整列表
            tag: 标签类型
            
        返回:
            根据所选标签筛选的文件子集
        '''
        _cnt_dict = {}
        for _filename in file_list:
            if tag == 'all':
                _ind = 0
            else:
                _ind = int(re.findall(r"(?<=room)\d+", str(_filename))[0])
            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        """
        计算SELD评估结果
        
        参数:
            pred_files_path: 预测文件路径
            is_jackknife: 是否使用jackknife方法
            
        返回:
            ER: 错误率
            F: F分数
            LE: 定位误差
            LR: 定位召回率
            seld_scr: SELD分数
            classwise_results: 每个类别的结果
        """
        # 收集预测文件信息
        pred_file_list = sorted(Path(pred_files_path).glob('*.csv'))
        pred_labels_dict = {}
        eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh)
        
        # 计算每个文件的分数
        for pred_cnt, pred_file in enumerate(pred_file_list):
            fn = pred_file.stem
            pred_dict = load_output_format_file(pred_file)
            pred_labels = to_metrics_format(pred_dict, self._ref_labels[fn][1], label_resolution=0.1)
            eval.update_seld_scores(pred_labels, self._ref_labels[fn][0])
            if is_jackknife:
                pred_labels_dict[fn] = pred_labels

        # 计算总体SED和DOA分数
        metric_dict, classwise_results = eval.compute_seld_scores(average=self._average)
        ER, F, LE, LR, seld_scr = list(metric_dict.values())

        if is_jackknife:
            # 使用jackknife方法计算置信区间
            global_values = [ER, F, LE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            
            # 使用留一法计算部分估计值
            for leave_file in pred_file_list:
                leave_one_out_list = pred_file_list[:]
                leave_one_out_list.remove(leave_file)
                eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    fn = pred_file.stem
                    eval.update_seld_scores(pred_labels_dict[fn], self._ref_labels[fn][0])
                metric_dict, classwise_results = eval.compute_seld_scores(average=self._average)
                ER, F, LE, LR, seld_scr = list(metric_dict.values())
                leave_one_out_est = [ER, F, LE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)
                    
            # 计算每个指标的jackknife统计量
            estimate, bias, std_err, conf_interval = [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                           global_value=global_values[i],
                           partial_estimates=partial_estimates[:, i],
                           significance_level=0.05
                           )
            return [ER, conf_interval[0]], [F, conf_interval[1]], [LE, conf_interval[2]], [LR, conf_interval[3]], [seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5,13,2) if len(classwise_results) else []]
      
        else: 
            return ER, F, LE, LR, seld_scr, classwise_results
    
    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
        获取所有类别的结果
        
        参数:
            pred_files_path: 预测文件路径
            score_type_list: 支持的评分类型
                'all' - 所有预测文件
                'room' - 单个房间的结果
        '''
        # 收集预测文件信息
        pred_file_list = sorted(Path(pred_files_path).glob('*.csv'))
        nb_pred_files = len(pred_file_list)

        print('预测文件数量: {}\n参考文件数量: {}'.format(nb_pred_files, self._nb_ref_files))

        for score_type in score_type_list:
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format('总分' if score_type=='all' else '每个{}的分数'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            # 收集对应score_type的文件
            split_cnt_dict = self.get_nb_files(pred_file_list, tag=score_type)
            
            # 计算给定score_type的分数
            for split_key in np.sort(list(split_cnt_dict)):
                eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh)
                samples_per_class = [0] * self._nb_classes
                
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    fn = pred_file.stem
                    pred_dict = load_output_format_file(pred_file)
                    pred_labels = to_metrics_format(pred_dict, self._ref_labels[fn][1], label_resolution=0.1)

                    # 统计每个房间中每个类别的样本数
                    for frame_ind in self._ref_labels[fn][2].keys():
                        for event in self._ref_labels[fn][2][frame_ind]:
                            samples_per_class[event[0]] += 1

                    eval.update_seld_scores(pred_labels, self._ref_labels[fn][0])

                # 计算总体SED和DOA分数
                metric_dict, classwise_test_scr = eval.compute_seld_scores(average=self._average)
                ER, F, LE, LR, seld_scr = list(metric_dict.values())

                print('\n{} {}数据的平均分数 (使用{}坐标)'.format(score_type, 'fold' if score_type=='all' else split_key, '极坐标' if self._use_polar_format else '笛卡尔坐标'))
                print('SELD分数 (早停指标): {:0.3f}'.format(seld_scr))
                print('SED指标: 错误率: {:0.3f}, F分数:{:0.1f}'.format(ER, 100*F))
                print('DOA指标: 定位误差: {:0.1f}, 定位召回率: {:0.1f}'.format(LE, 100*LR))
                print('{}中每个类别的样本数: {}'.format('所有房间' if score_type=='all' else '房间 ' + str(split_key), samples_per_class))
                
                if self._average == 'macro':
                    for cls_cnt in range(len(classwise_test_scr[0])):
                        words = '{:.0f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{}'.format(
                            classwise_test_scr[-1][cls_cnt], 
                            classwise_test_scr[0][cls_cnt], 
                            classwise_test_scr[1][cls_cnt], 
                            classwise_test_scr[2][cls_cnt],
                            classwise_test_scr[3][cls_cnt], 
                            classwise_test_scr[4][cls_cnt], 
                            samples_per_class[int(classwise_test_scr[-1][cls_cnt])])
                        print(words)


def reshape_3Dto2D(A):
    """将3D数组重塑为2D数组"""
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def main():
    """主函数"""
    # 设置空间阈值和平均方式
    spatial_threshold = 20
    average = 'macro'

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument('--use_jackknife', action='store_true', help='使用jackknife方法')
    parser.add_argument('--consolidated_score', action='store_true', help='计算综合SELD分数')
    parser.add_argument('--gt_csv_dir', type=str, help='真实标签CSV目录')
    parser.add_argument('--pred_csv_dir', type=str, help='预测结果CSV目录')
    parser.add_argument('--nb_classes', default=12, type=int, help='类别数量')
    args = parser.parse_args()
    
    use_jackknife = args.use_jackknife
    score_obj = ComputeSELDResults(ref_files_folder=args.gt_csv_dir, nb_classes=args.nb_classes, 
                                   doa_thresh=spatial_threshold, average=average)

    # 计算DCASE最终结果
    if not use_jackknife:
        if not args.consolidated_score:
            # 计算宏平均结果
            score_obj._average = 'macro'
            ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(args.pred_csv_dir)
            print('#### 未见测试数据的类别结果 ####')
            words = '类别\tER\tF\tLE\tLR\tSELD分数'
            print(words)
            
            # 打印每个类别的结果
            for cls_cnt in range(args.nb_classes):
                words = '{}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'\
                    .format(cls_cnt, classwise_test_scr[0][cls_cnt], classwise_test_scr[1][cls_cnt], 
                           classwise_test_scr[2][cls_cnt], classwise_test_scr[3][cls_cnt], 
                           classwise_test_scr[4][cls_cnt])
                print(words)
                
            words = '宏平均\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(ER, F, LE, LR, seld_scr)
            print('######## 宏平均 ########')
            print('SELD分数 (早停指标): {:0.3f}'.format(seld_scr))
            print('SED指标: 错误率: {:0.3f}, F分数:{:0.1f}'.format(ER, 100*F))
            print('DOA指标: 定位误差: {:0.1f}, 定位召回率: {:0.1f}'.format(LE, 100*LR))
            
            # 计算微平均结果
            score_obj._average = 'micro'
            ER, F, LE, LR, seld_scr, _ = score_obj.get_SELD_Results(args.pred_csv_dir)
            words = '微平均\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(ER, F, LE, LR, seld_scr)
            print('######## 微平均 ########')
            print('SELD分数 (早停指标): {:0.3f}'.format(seld_scr))
            print('SED指标: 错误率: {:0.3f}, F分数:{:0.1f}'.format(ER, 100*F))
            print('DOA指标: 定位误差: {:0.1f}, 定位召回率: {:0.1f}'.format(LE, 100*LR))
        else:
            # 计算综合结果
            score_obj.get_consolidated_SELD_results(args.pred_csv_dir)
    else:
        # 使用jackknife方法计算结果
        ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(args.pred_csv_dir,is_jackknife=use_jackknife )
        print('SELD分数 (早停指标): {:0.3f} {}'.format(seld_scr[0], '[{:0.3f}, {:0.3f}]'.format(seld_scr[1][0], seld_scr[1][1]) ))
        print('SED指标: 错误率: {:0.3f} {}, F分数: {:0.1f} {}'.format(ER[0] , '[{:0.3f},  {:0.3f}]'\
            .format(ER[1][0], ER[1][1]) , 100*F[0], '[{:0.3f}, {:0.3f}]'.format(100*F[1][0], 100*F[1][1]) ))
        print('DOA指标: 定位误差: {:0.1f} {}, 定位召回率: {:0.1f} {}'\
            .format(LE[0], '[{:0.3f}, {:0.3f}]'.format(LE[1][0], LE[1][1]) , 100*LR[0],'[{:0.3f}, {:0.3f}]'.format(100*LR[1][0], 100*LR[1][1]) ))
        print('未见测试数据的类别结果')
        print('类别\tER\tF\tLE\tLR\tSELD分数')
        for cls_cnt in range(args.nb_classes):
            print('{}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}'.format(
                cls_cnt, 
                classwise_test_scr[0][0][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]), 
                classwise_test_scr[0][1][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]), 
                classwise_test_scr[0][2][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) , 
                classwise_test_scr[0][3][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) , 
                classwise_test_scr[0][4][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1])))


if __name__ == "__main__":
    main()